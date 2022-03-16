import torch
from transformers import AdamW
from dataset import MyDataset, MyDataset_triples, BinaryDataset
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from utils import compute_acc
from colbert import ColBERT
import os
from transformers import logging
logging.set_verbosity_warning()
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
pretrained_model = 'hfl/chinese-macbert-base'  
lr = 1e-5
batch_size = 4
accumulation_steps = 1
mode = 'train'
epochs = 3
warm_up_rate = 0.03
json_path = './data/train_complete.json'
train_pairs_path = './data/train_binary_pairs'
train_negative_nums = 10
multi_gpu = False
warm_up = True
model_save_path = './outputs_models/binary/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
### hyperparams ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)
if device == 'cpu':
    multi_gpu = False


json_data = load_data_json(json_path)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
# train_set = MyDataset(mode, json_data, tokenizer, train_negative_nums)
train_set = BinaryDataset(mode, json_data, tokenizer,  train_pairs_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')




colbert = ColBERT(pretrained_model, device=device)
colbert_model_path = './outputs_models/colbert_neg150/model_2.pt'
colbert.load_state_dict(torch.load(colbert_model_path), strict=False)


model = MyModel(pretrained_model)

model_params = model.named_parameters()
model_dict_params = dict(model_params)

colbert_params = colbert.named_parameters()
colbert_dict_params = dict(colbert_params)

for name, param in colbert_dict_params.items():
    if 'bert' in name:
        model_dict_params[name].data.copy_(param.data)
        model_dict_params[name].requires_grad = False
    
model.load_state_dict(model_dict_params, strict=False)

optimizer = AdamW(model.parameters(), lr=lr)
if warm_up:
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)


class_weight = torch.FloatTensor([1/10561, 1/1383]).to(device)
loss_fct = nn.CrossEntropyLoss(weight=class_weight)
# loss_fct = nn.CrossEntropyLoss()

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    for i, data in enumerate(train_loader):        
        test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask, labels = [t.to(device) for t in data]

        D_test = (test_input_ids, test_attention_mask)
        D_ref = (ref_input_ids, ref_attention_mask)
        # forward pass
        outputs = model(D_test, D_ref)


        loss = loss_fct(outputs, labels)
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()

        if ((i+1) % accumulation_steps) or ((i+1) == len(train_loader)) == 0:
            optimizer.step()
            optimizer.zero_grad()
            if warm_up:
                scheduler.step()
        acc += compute_acc(outputs, labels)

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )

    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')