import string
import torch
import torch.nn as nn
from zhon import hanzi
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, AutoConfig, AutoModel


class ColBERT(BertPreTrainedModel):
    def __init__(self, PRETRAINED_LM, device='cpu', doc_maxlen=512, mask_punctuation=True, dim=128, similarity_metric='l2'):

        super(ColBERT, self).__init__(config=AutoConfig.from_pretrained(PRETRAINED_LM))

        self.config = AutoConfig.from_pretrained(PRETRAINED_LM)
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric

        self.cos = nn.CosineSimilarity(dim=2)
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        
        if self.mask_punctuation:
            self.skiplist = {}
            punctuation = string.punctuation + hanzi.punctuation
            self.tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_LM)
            for symbol in punctuation:
                if symbol ==  '　':
                    continue
                self.skiplist[self.tokenizer.encode(symbol, add_special_tokens=False)[0]] = True


        self.bert = AutoModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True)
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=False)


        self.init_weights()

    def forward(self, D_test, D_ref):
        # test_input_ids, test_attention_mask = D_test
        # ref_input_ids, ref_attention_mask = D_ref
        test_input_ids, test_attention_mask, test_token_type_ids = D_test
        ref_input_ids, ref_attention_mask, ref_token_type_ids = D_ref

        return self.score(
            self.doc(test_input_ids, test_attention_mask, test_token_type_ids), 
            self.doc(ref_input_ids, ref_attention_mask, ref_token_type_ids)
        )


    def doc(self, input_ids, attention_mask, token_type_ids, keep_dims=True):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # Bsize * Length * Dim=128 
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        return D


    def score(self, Q, D):

        if Q.size(0) != D.size(0):
            Q = torch.repeat_interleave(Q, 2, dim=0)

        if self.similarity_metric == 'cosine':
            # print('cosine')
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask