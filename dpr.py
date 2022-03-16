import string
import torch
import torch.nn as nn
from zhon import hanzi
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, AutoConfig, AutoModel
import torch.nn.functional as F




def cal_nll_loss(scores, positive_idx_per_question):
    softmax_scores = F.log_softmax(scores, dim=1)
    loss = F.nll_loss(
            softmax_scores,
            positive_idx_per_question,
            reduction="mean",
        )
    max_score, max_idxs = torch.max(softmax_scores, 1)
    correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
    return loss, correct_predictions_count

class DPR(BertPreTrainedModel):
    def __init__(self, PRETRAINED_LM):

        super(DPR, self).__init__(config=AutoConfig.from_pretrained(PRETRAINED_LM))

        self.config = AutoConfig.from_pretrained(PRETRAINED_LM)
        self.bert = AutoModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True)
        self.init_weights()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, D_test, D_ref):
        test_input_ids, test_attention_mask = D_test
        ref_input_ids, ref_attention_mask = D_ref

        return self.score(
            self.doc(test_input_ids, test_attention_mask), 
            self.doc(ref_input_ids, ref_attention_mask)
        )


    def doc(self, input_ids, attention_mask, keep_dims=True):
        D = self.bert(input_ids, attention_mask=attention_mask)[1]
        return D

    def score(self, Q, D):


        scores = F.cosine_similarity(Q.unsqueeze(1), D.unsqueeze(0), dim=2)

        return scores
