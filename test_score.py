import torch
from transformers import AdamW, BertForSequenceClassification
from dataset import MyDataset, TestDataset
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
from dpr import DPR
import pickle

### hyperparams ###
# pretrained_model = 'hfl/chinese-bert-wwm'
pretrained_model = 'hfl/chinese-macbert-base'  

mode = 'test'
data_mode = 'public'
json_path = f'./data/{data_mode}_complete.json'
batch_size = 16

result_path = f'./output_results/{data_mode}_result_colbert_text_keywords_hard_neg5_rand_neg30_epoch{score_used_epoch}'
### hyperparams ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)

json_data = load_data_json(json_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

test_set = MyDataset(mode, json_data, tokenizer, add_marker=False)
print(len(test_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# model_paths  = [
#     f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble1/model_{score_used_epoch}.pt',
#     f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble2/model_{score_used_epoch}.pt',
#     f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble3/model_{score_used_epoch}.pt',
#     f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble4/model_{score_used_epoch}.pt',
#     f'./outputs_models/cross_encoder_keywords_title_text_large_ensemble5/model_{score_used_epoch}.pt',
# ]

score_used_epoch = 2
score_model_path = f'./outputs_models/colbert_text_keywords_hard_neg5_rand_neg30/model_{score_used_epoch}.pt'

score_model = ColBERT(pretrained_model, device=device)
# score_model = DPR(pretrained_model)
score_model.load_state_dict(torch.load(score_model_path), strict=False)
score_model = score_model.to(device)
score_model = score_model.eval()




# pred_list = []

results = {}
all_test_dids = []
all_ref_dids = []
all_score_preds = []



with torch.no_grad():
    for i , data in enumerate(tqdm(test_loader)):
        # test_input_ids, test_attention_mask, ref_input_ids, ref_attention_mask = [t.to(device) for t in data[0]]
        test_input_ids, test_attention_mask, test_token_type_ids = [t.to(device) for t in data[0]]
        ref_input_ids , ref_attention_mask, ref_token_type_ids = [t.to(device) for t in data[1]]
        test_dids, ref_dids = data[2]
        all_test_dids += list(test_dids)
        all_ref_dids += list(ref_dids)

        D_test = (test_input_ids, test_attention_mask, test_token_type_ids)
        D_ref = (ref_input_ids, ref_attention_mask, ref_token_type_ids)

        score_outputs = score_model(D_test, D_ref)
        score_preds = score_outputs.detach().cpu().numpy()
        # score_preds = score_outputs.detach().cpu().numpy()[0]
        all_score_preds += list(score_preds)


for i in range(len(all_score_preds)):
    test_did = all_test_dids[i]
    ref_did = all_ref_dids[i]
    score_pred = all_score_preds[i]
    if test_did in results:
        results[test_did].append((ref_did, score_pred))
    else:
        results[test_did] = [(ref_did, score_pred)]

with open(result_path, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

print(result_path)

