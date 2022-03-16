from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, precision_score

def compute_acc(pred, label):
    pred = pred.detach()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return (pred == label).float().mean(), (pred != label).cpu().numpy()

def compute_f1(pred, label):
    pred = pred.detach().cpu()
    label = label.cpu()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return f1_score(label, pred, zero_division=1)

def compute_recall(pred, label):
    pred = pred.detach().cpu()
    label = label.cpu()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return recall_score(label, pred, zero_division=1)

def compute_precision(pred, label):
    pred = pred.detach().cpu()
    label = label.cpu()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return precision_score(label, pred, zero_division=1)


def compute_all(pred, label):
    pred = pred.detach().cpu()
    label = label.cpu()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, zero_division=1)
    return precision, recall, f1