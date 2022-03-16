import torch
from transformers import AdamW, BertForSequenceClassification
from dataset import MyDataset, MyDataset_triples, BinaryDataset, CrossEncoderDataset
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import compute_acc, compute_recall, compute_f1, compute_all, compute_precision
from torch.utils.data import random_split
from colbert import ColBERT
import os
import numpy as np
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_warning()
### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'  
# pretrained_model = 'hfl/chinese-roberta-wwm-ext-large'
pretrained_model = 'hfl/chinese-macbert-large'  
lr = 1e-5
batch_size = 1
val_batch_size = 2
accumulation_steps = 32
mode = 'train'
epochs = 2
warm_up_rate = 0.05
json_path = './data/train_complete.json'
train_json_path = './data/mytrain.json'
val_json_path = './data/myval.json'

# train_pairs_path = './data/train_binary_pairs'
train_hard_negative_nums = 10
train_rand_negative_nums = 15
multi_gpu = False
warm_up = True
valid = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)
if device == 'cpu':
    multi_gpu = False
json_data = load_data_json(json_path)


### hyperparams ###

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
ensemble_nums = 3
for k in range(0, ensemble_nums):
    pretrain_model_name_path = pretrained_model.split('/')[-1]
    model_name = f'{pretrain_model_name_path}_hard10_rand20_title_and_text_fixkeywords_alldata_ensemble{k+1}'
    print(model_name)
    model_save_path = f'./outputs_models/{model_name}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    log_fp = open(f'./logs/{model_name}.txt', 'w')
    print(f'ensemble {k+1}', file=log_fp)

    # ##### here for train
    # train_json_data = load_data_json(train_json_path)
    # valid_json_data = load_data_json(val_json_path)
    train_set = CrossEncoderDataset(mode, json_data, tokenizer, train_hard_negative_nums, train_rand_negative_nums, train_pairs_path=None)
    print(len(train_set))
    # Random split
    if valid:
        train_set_size = int(len(train_set) * 0.9)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])
        print(f'train_size : {train_set_size}, val_size {valid_set_size}')
        valid_loader = DataLoader(valid_set, batch_size=val_batch_size, shuffle=False)

    else:
        train_set_size = len(train_set)
        valid_set_size = 0
    

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_steps = len(train_loader) * epochs / (batch_size * accumulation_steps)
    warm_up_steps = total_steps * warm_up_rate
    print(f'warm_up_steps : {warm_up_steps}')
    model = BertForSequenceClassification.from_pretrained(pretrained_model)
    optimizer = AdamW(model.parameters(), lr=lr)

    if warm_up:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_steps)
    class_weight = torch.FloatTensor([100/10000, 100/1300]).to(device)
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
        recall = 0.0
        f1 = 0.0
        precision = 0.0
        model.train()
        for i, data in enumerate(train_loader):        
            input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data[:-1]]

            # test_dids , ref_dids = data[-1]
            # test_dids = np.array(test_dids)
            # ref_dids = np.array(ref_dids)


            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            logits = outputs.logits
            # preds = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1]

            loss = loss_fct(logits, labels)
            running_loss += loss.item()
            loss = loss / accumulation_steps

            loss.backward()
            if ((i+1) % accumulation_steps) == 0 or ((i+1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if warm_up:
                    scheduler.step()
            
            batch_acc , wrong_idx = compute_acc(logits, labels)
            # wrong_test_dids = test_dids[wrong_idx]
            # wrong_ref_dids = ref_dids[wrong_idx]
            # wrong_labels = labels[wrong_idx]
            # wrong_preds = preds[wrong_idx]
            # for wrong_test_did, wrong_ref_did, wrong_label, wrong_pred in zip(wrong_test_dids, wrong_ref_dids, wrong_labels, wrong_preds):
            #     print(f'test : {wrong_test_did}, ref : {wrong_ref_did}, label : {wrong_label}, pred : {wrong_pred}')
            acc += batch_acc
            precision += compute_precision(logits, labels)
            recall += compute_recall(logits, labels)
            f1 += compute_f1(logits, labels)
            print(f'\r Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}, precision {precision/ (i+1) :.5f} recall {recall/ (i+1) :.5f} f1 {f1/ (i+1) :.5f}' , end='' )
        print(f'Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}, precision {precision/ (i+1) :.5f} recall {recall/ (i+1) :.5f} f1 {f1/ (i+1) :.5f}' , file=log_fp)
        print('')
        # valid 
        if valid:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            totals_batch = len(valid_loader)
            val_recall = 0.0
            val_f1 = 0.0
            val_precision = 0.0
            for i, data in enumerate(valid_loader):        
                input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in data[:-1]]
                test_dids , ref_dids = data[-1]
                test_dids = np.array(test_dids)
                ref_dids = np.array(ref_dids)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                    logits = outputs.logits

                    loss = loss_fct(logits, labels)
                    preds = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1]

                val_loss += loss.item()
                batch_acc , wrong_idx = compute_acc(logits, labels)
                wrong_test_dids = test_dids[wrong_idx]
                wrong_ref_dids = ref_dids[wrong_idx]
                wrong_labels = labels[wrong_idx]
                wrong_preds = preds[wrong_idx]
                for wrong_test_did, wrong_ref_did, wrong_label, wrong_pred in zip(wrong_test_dids, wrong_ref_dids, wrong_labels, wrong_preds):
                    print(f'test : {wrong_test_did}, ref : {wrong_ref_did}, label : {wrong_label}, pred : {wrong_pred :.3f}', file=log_fp)

                val_acc += batch_acc
                val_precision += compute_precision(logits, labels)
                val_recall += compute_recall(logits, labels)
                val_f1 += compute_f1(logits, labels)

                
                print(f'\r[val]Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}, precision {val_precision/ (i+1) :.5f} recall {val_recall/ (i+1) :.5f} f1 {val_f1/ (i+1) :.5f}' , end='' )
        if valid:
            print(f'[val]Epoch : {epoch+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}, precision {val_precision/ (i+1) :.5f} recall {val_recall/ (i+1) :.5f} f1 {val_f1/ (i+1) :.5f}', file=log_fp)
        log_fp.flush()
        torch.save(model.state_dict(), f"{model_save_path}/model_{str(epoch+1)}.pt")
        print(' saved ')
    log_fp.close()