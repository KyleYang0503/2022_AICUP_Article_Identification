import torch
from transformers import AdamW
from dataset import MyDataset, MyDataset_triples
from model import MyModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from utils import compute_acc
from colbert import ColBERT
from dpr import DPR, cal_nll_loss
import os
import numpy as np
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
pretrained_model = 'hfl/chinese-macbert-large'  
lr = 1e-5
batch_size = 12
accumulation_steps = 1
mode = 'train'
epochs = 20
warm_up_rate = 0.03
json_path = './data/train_complete.json'
train_hard_negative_nums = 2
train_rand_negative_nums = 10
multi_gpu = False
model_save_path = './outputs_models/dpr_title_keyword/'
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
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
warm_up_steps = total_steps * warm_up_rate
print(f'warm_up_steps : {warm_up_steps}')


model = DPR(pretrained_model)


optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_steps)

model = model.to(device)
if multi_gpu:   
    model = nn.DataParallel(model)
model.train()


labels = range(0, 64, 2)

for epoch in range(epochs):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    for i, data in enumerate(train_loader):        
        # test_input_ids, test_attention_mask, test_token_type_ids, ref_input_ids, ref_attention_mask, ref_token_type_ids = [t.to(device) for t in data]
        test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask = [t.to(device) for t in data]

        

        ref_input_ids = ref_input_ids.view(ref_input_ids.size(0) * ref_input_ids.size(1), ref_input_ids.size(-1))
        ref_attention_mask = ref_attention_mask.view(ref_attention_mask.size(0) * ref_attention_mask.size(1), ref_attention_mask.size(-1))
        # ref_token_type_ids = ref_token_type_ids.view(ref_token_type_ids.size(0) * ref_token_type_ids.size(1), ref_token_type_ids.size(-1))
        
        
        # forward pass
        D_test = (test_input_ids, test_attention_mask)
        D_ref = (ref_input_ids, ref_attention_mask)

        scores = model(D_test, D_ref)
        b = scores.size(0)
        positive_idx_per_question = torch.tensor(labels[:b]).to(device)

        loss, correct_predictions_count = cal_nll_loss(scores, positive_idx_per_question)


        # loss = loss_fct(outputs, labels[:outputs.size(0)])
        running_loss += loss.item()
        loss = loss / accumulation_steps

        loss.backward()

        if ((i+1) % accumulation_steps) or ((i+1) == len(train_loader)) == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        acc += correct_predictions_count.detach().cpu().item() / b

        print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )


    torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
    print(' saved ')