from transformers import AutoModel, AutoConfig, BertPreTrainedModel
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import ACT2FN


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class MyModel(BertPreTrainedModel):
    def __init__(self, PRETRAINED_LM):
        super(MyModel, self).__init__(config=AutoConfig.from_pretrained(PRETRAINED_LM))
        self.config = AutoConfig.from_pretrained(PRETRAINED_LM)
        self.bert = AutoModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True)
        self.linear = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, 2)
            # nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(self.config.hidden_size, 2)
        )
        
        self.pooler = BertPooler(self.config)
        self.attention =  nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads =self.config.num_attention_heads,
            dropout=self.config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.init_weights()

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    
        
    def forward(self, D_test, D_ref):
        test_input_ids, test_attention_mask = D_test
        ref_input_ids, ref_attention_mask = D_ref
        
        test_outputs = self.bert(
            test_input_ids,
            attention_mask=test_attention_mask)

        ref_outputs = self.bert(
            ref_input_ids,
            attention_mask=ref_attention_mask)

        ### cat last hidden layer with attention to predict
        ###
        ###
        # test_hidden = test_outputs[0]
        # ref_hidden = ref_outputs[0]
        # cat_hidden  = torch.cat((test_hidden, ref_hidden) , dim=1)
        # attn_output, attn_output_weights = self.attention(cat_hidden, cat_hidden, cat_hidden)
        # layer_output = self.feed_forward(attn_output)
        # pooled_output = self.pooler(layer_output)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)


        ### cat CLS to predict 
        ###
        ###
        # test_cls = test_outputs[1]
        # ref_cls = ref_outputs[1]
        # concat_cls = torch.cat((test_cls, ref_cls) , dim=-1)
        # logits = self.linear(concat_cls)

        ### pooling use max token 
        ###
        ###
        # test_hidden = test_outputs[0]
        # ref_hidden = ref_outputs[0]
        # test_max, _ = torch.max(test_hidden, dim=1)
        # ref_max, _ = torch.max(ref_hidden, dim=1)
        # cat_max = torch.cat((test_max, ref_max) , dim=-1)
        # logits = self.linear(cat_max)


        ### pooling use mean token
        ###
        ###
        test_hidden = test_outputs[0]
        test_input_mask_expanded = test_attention_mask.unsqueeze(-1).expand(test_hidden.size()).float()
        test_mean = torch.sum(test_hidden * test_input_mask_expanded, 1) / torch.clamp(test_input_mask_expanded.sum(1), min=1e-9)

        ref_hidden = ref_outputs[0]
        ref_input_mask_expanded = ref_attention_mask.unsqueeze(-1).expand(ref_hidden.size()).float()
        ref_mean = torch.sum(ref_hidden * ref_input_mask_expanded, 1) / torch.clamp(ref_input_mask_expanded.sum(1), min=1e-9)
        # cat_mean = torch.cat((test_mean, ref_mean)) , dim=-1)

        cat_mean = torch.cat((test_mean, ref_mean, torch.abs(torch.sub(test_mean, ref_mean))) , dim=-1)
        logits = self.linear(cat_mean)




        ### first layer + last layer and concat to attention
        ###
        ###
        # test_first_layer_hidden = test_outputs[-1][1]
        # test_last_layer_hidden = test_outputs[-1][-1]
        # test_first_add_last = torch.add(test_first_layer_hidden, test_last_layer_hidden)
        # ref_first_layer_hidden = ref_outputs[-1][1]
        # ref_last_layer_hidden = ref_outputs[-1][-1]
        # ref_first_add_last = torch.add(ref_first_layer_hidden, ref_last_layer_hidden)
        # cat_hidden  = torch.cat((test_first_add_last, ref_first_add_last) , dim=1)
        # attn_output, attn_output_weights = self.attention(cat_hidden, cat_hidden, cat_hidden)
        # layer_output = self.feed_forward(attn_output)
        # pooled_output = self.pooler(layer_output)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)


        ### first layer + last layer and get max to linear
        ###
        ###
        # test_first_layer_hidden = test_outputs[-1][1]
        # test_last_layer_hidden = test_outputs[-1][-1]
        # test_first_add_last = torch.add(test_first_layer_hidden, test_last_layer_hidden)
        # ref_first_layer_hidden = ref_outputs[-1][1]
        # ref_last_layer_hidden = ref_outputs[-1][-1]
        # ref_first_add_last = torch.add(ref_first_layer_hidden, ref_last_layer_hidden)
        # test_max, _ = torch.max(test_first_add_last, dim=1)
        # ref_max, _ = torch.max(ref_first_add_last, dim=1)
        # cat_max = torch.cat((test_max, ref_max) , dim=-1)
        # logits = self.linear(cat_max)

        return logits


