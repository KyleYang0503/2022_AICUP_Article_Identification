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
# pretrained_model = 'hfl/chinese-bert-wwm'
pretrained_model = 'hfl/chinese-macbert-large'  
# pretrained_model = 'hfl/chinese-roberta-wwm-ext-large'

mode = 'test'
data_mode = 'public'
json_path = f'./data/{data_mode}_complete.json'
# val_json_path = './data/myval.json'
batch_size = 75
# model
# binary_used_epoch = 2
# binary_model_path = f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble5/model_{binary_used_epoch}.pt'

# result_path = f'./output_results/{data_mode}_binary_result_cross_encoder_keywords_title_text_large_ensemble_5models_epoch{binary_used_epoch}'
# print(result_path)
multi_gpu = False
### hyperparams ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)

json_data = load_data_json(json_path)
# val_json_data = load_data_json(val_json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
test_set = CrossEncoderDataset(mode, json_data, tokenizer)
# test_set = CrossEncoderDataset(mode, val_json_data, tokenizer)
print(len(test_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


binary_used_epoch = 2
model_paths  = [
    # f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble_alldata1/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble_alldata2/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble_alldata3/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble_alldata4/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_20_40_keywords_title_text_large_ensemble1/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_20_40_keywords_title_text_large_ensemble2/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_20_40_keywords_title_text_large_ensemble3/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_20_40_keywords_title_text_large_ensemble4/model_{binary_used_epoch}.pt',
    # f'./outputs_models/cross_encoder_20_40_keywords_title_text_large_ensemble5/model_{binary_used_epoch}.pt',

    f'./outputs_models/chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble1/model_{binary_used_epoch}.pt',
    f'./outputs_models/chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble2/model_{binary_used_epoch}.pt',
    f'./outputs_models/chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble3/model_{binary_used_epoch}.pt',



]
results_paths = [
    # f'./output_results/{data_mode}_binary_result_cross_encoder_keywords_title_text_large_ensemble_alldata1',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_keywords_title_text_large_ensemble_alldata2',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_keywords_title_text_large_ensemble_alldata3',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_keywords_title_text_large_ensemble_alldata4',

    # f'./output_results/{data_mode}_binary_result_cross_encoder_20_40_keywords_title_text_large_ensemble1',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_20_40_keywords_title_text_large_ensemble2',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_20_40_keywords_title_text_large_ensemble3',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_20_40_keywords_title_text_large_ensemble4',
    # f'./output_results/{data_mode}_binary_result_cross_encoder_20_40_keywords_title_text_large_ensemble5',

    f'./output_results/{data_mode}_binary_result_chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble1',
    f'./output_results/{data_mode}_binary_result_chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble2',
    f'./output_results/{data_mode}_binary_result_chinese-macbert-large_hard20_rand0_title_and_text_fixkeywords_ensemble3',

]

# def load_models(model_paths):
#     models = []
#     for model_path in model_paths:
#         # load binary model cross encoder
#         binary_model = BertForSequenceClassification.from_pretrained(pretrained_model)
#         binary_model.load_state_dict(torch.load(model_path), strict=False)
#         # binary_model = binary_model.to(device)
#         binary_model = binary_model.eval()
#         models.append(binary_model)
#     return models

# models = load_models(model_paths)
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
