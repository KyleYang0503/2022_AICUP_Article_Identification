import torch
from transformers import AdamW, BertForSequenceClassification
from dataset import MyDataset, TestDataset, CrossEncoderDataset
from model import MyModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from preprocess import load_data_json
from transformers import AutoTokenizer
from utils import compute_acc
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from colbert import ColBERT
import pickle

### hyperparams ###
pretrained_model = 'hfl/chinese-macbert-large'  

mode = 'test'
data_mode = 'public'
json_path = f'./data/{data_mode}_complete.json'
batch_size = 75
multi_gpu = False
### hyperparams ###
device = torch.device('cuda')
print("device:", device)

json_data = load_data_json(json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
test_set = CrossEncoderDataset(mode, json_data, tokenizer)
print(len(test_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


binary_used_epoch = 2
model_paths  = [
    
    './outputs_models/chinese-macbert-large_hard5_rand5_title_and_text_fixkeywords_alldata_ensemble1',
    './outputs_models/chinese-macbert-large_hard5_rand5_title_and_text_fixkeywords_alldata_ensemble2',
    './outputs_models/chinese-macbert-large_hard5_rand5_title_and_text_fixkeywords_alldata_ensemble3',

    './outputs_models/chinese-macbert-large_hard2_rand3_title_and_text_fixkeywords_ensemble1',
    './outputs_models/chinese-macbert-large_hard2_rand3_title_and_text_fixkeywords_ensemble2',
    './outputs_models/chinese-macbert-large_hard2_rand3_title_and_text_fixkeywords_ensemble3',

    './outputs_models/chinese-macbert-large_hard4_rand6_alldata_ensemble1',
    './outputs_models/chinese-macbert-large_hard4_rand6_alldata_ensemble2',
    './outputs_models/chinese-macbert-large_hard4_rand6_alldata_ensemble3',

]

results_paths = [f'./final_results/{x.split("/")[-1]}' for x in model_paths]
model_paths = [x + '/model_2.pt' for x in model_paths]


num_of_models = len(model_paths)

for model_path , result_path in zip(model_paths, results_paths):
    binary_model = BertForSequenceClassification.from_pretrained(pretrained_model)
    binary_model.load_state_dict(torch.load(model_path), strict=False)
    binary_model = binary_model.eval()
    binary_model = binary_model.to(device)
    if multi_gpu:
        binary_model = nn.DataParallel(binary_model)
    results = {}
    all_test_dids = []
    all_ref_dids = []
    all_binary_preds = []
    with torch.no_grad():
        for i , data in enumerate(tqdm(test_loader)):
            input_ids, attention_mask, token_type_ids = [t.to(device) for t in data[0]]
            test_dids, ref_dids = data[1]
            all_test_dids += list(test_dids)
            all_ref_dids += list(ref_dids)
            binary_outputs = binary_model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            binary_outputs = binary_outputs.logits
            binary_preds = F.softmax(binary_outputs, dim=1).detach().cpu().numpy()[:, 1]
            all_binary_preds += list(binary_preds)

    for i in range(len(all_binary_preds)):
        test_did = all_test_dids[i]
        ref_did = all_ref_dids[i]
        binary_pred = all_binary_preds[i]
        if test_did in results:
            results[test_did].append((ref_did, binary_pred))
        else:
            results[test_did] = [(ref_did, binary_pred)]

    with open(result_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(result_path)
    binary_model = binary_model.to('cpu')
