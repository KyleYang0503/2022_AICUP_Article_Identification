import torch
from transformers import AdamW
from dataset import MyDataset, MyDataset_triples
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import compute_acc
from colbert import ColBERT
import os
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
pretrained_model = 'hfl/chinese-macbert-base'  
lr = 5e-5
batch_size = 2
accumulation_steps = 16
mode = 'train'
epochs = 3
warm_up_rate = 0.06
json_path = './data/train_complete.json'
train_hard_negative_nums = 10
train_rand_negative_nums = 30
multi_gpu = False
# model_save_path = f'./outputs_models/colbert_hard_neg{train_hard_negative_nums}_rand_neg{train_rand_negative_nums}/'
model_save_path = f'./outputs_models/colbert_text_keywords_hard2_rand3/'

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
train_set = MyDataset_triples(mode, json_data, tokenizer, train_hard_negative_nums, train_rand_negative_nums)
print(len(train_set))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')


model = ColBERT(pretrained_model, device=device)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_steps)

loss_fct = nn.CrossEntropyLoss()

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


labels = torch.zeros(batch_size, dtype=torch.long, device=device)

for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    for i, data in enumerate(train_loader):        
        # test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask = [t.to(device) for t in data]
        test_input_ids, test_attention_mask, test_token_type_ids = [t.to(device) for t in data[0]]
        ref_input_ids , ref_attention_mask, ref_token_type_ids = [t.to(device) for t in data[1]]
        b = test_input_ids.size(0)



        ref_input_ids = ref_input_ids.view(ref_input_ids.size(0) * ref_input_ids.size(1), ref_input_ids.size(-1))
        ref_attention_mask = ref_attention_mask.view(ref_attention_mask.size(0) * ref_attention_mask.size(1), ref_attention_mask.size(-1))
        ref_token_type_ids = ref_token_type_ids.view(ref_token_type_ids.size(0) * ref_token_type_ids.size(1), ref_token_type_ids.size(-1))

        

        # forward pass
        D_test = (test_input_ids, test_attention_mask, test_token_type_ids)
        D_ref = (ref_input_ids, ref_attention_mask, ref_token_type_ids)

        outputs = model(D_test, D_ref)

        outputs = outputs.view(-1, 2)


        loss = loss_fct(outputs, labels[:outputs.size(0)])
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()

        if ((i+1) % accumulation_steps) or ((i+1) == len(train_loader)) == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        batch_acc , _ = compute_acc(outputs, labels[:outputs.size(0)])
        acc += batch_acc

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )
    
    print('regenerate dataset ')
    train_set = MyDataset_triples(mode, json_data, tokenizer, train_hard_negative_nums, train_rand_negative_nums)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')