import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from transformers import BertModel, XLMRobertaModel, RobertaModel
#from transformers.modeling_bert import BertLayerNorm
BertLayerNorm = torch.nn.LayerNorm
def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights

class BertForQASimple(nn.Module):
    def __init__(self, config):
        super(BertForQASimple, self).__init__()
        self.bert = BertModel(config)
        self.qa_outputs  = nn.Linear(config.hidden_size, 2)
        self.cls_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, token_type_ids, attention_mask, doc_mask,
                start_positions=None, end_positions=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        span_logits = self.qa_outputs(sequence_output)

        doc_mask[:,0] = 0
        span_logits = span_logits + (1.0 - doc_mask.unsqueeze(-1)) * -10000.0
        start_logits, end_logits = span_logits.split(1, dim=-1)  # use_cls
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None:
            total_loss = 0
            # If we are on multi-GPU, split add a dimension - if not this is a no-op
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_noans = (start_positions == 0).float()
            loss_fct_span  = CrossEntropyLoss(ignore_index=ignored_index,reduction='none')
            #loss_fct_noans = BCEWithLogitsLoss()
            start_loss = (loss_fct_span(start_logits, start_positions)*(1-is_noans)).mean()
            end_loss   = (loss_fct_span(end_logits,   end_positions  )*(1-is_noans)).mean()
            #cls_loss   = loss_fct_noans(cls_logits, is_noans)
            total_loss += (start_loss + end_loss)/2 #+ cls_loss
            return start_logits, end_logits, total_loss
        else:
            return start_logits, end_logits

class XLMRForGLUESimple(nn.Module):
    def __init__(self, config, num_labels):
        super(XLMRForGLUESimple, self).__init__()

        config.num_labels = num_labels
        self.num_labels = num_labels
        config.output_hidden_states = True
        config.output_attentions = False
        self.roberta = XLMRobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, outputs_for_caching=False,**kwargs):
        head_mask = kwargs.pop('head_mask',None)
        last_hidden_state, pooled_output, hidden_states = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, head_mask=head_mask)
        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels
        #assert len(sequence_output)==self.bert.config.num_hidden_layers + 1  # embeddings + 12 hiddens
        #assert len(attention_output)==self.bert.config.num_hidden_layers + 1 # None + 12 attentions
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits,labels)
            return logits, hidden_states, loss
        else:
            if outputs_for_caching is True:
                return logits, hidden_states
            else:
                return logits

            
            
class RoBERTaForGLUESimple(nn.Module):
    def __init__(self, config, num_labels):
        super(RoBERTaForGLUESimple, self).__init__()

        config.num_labels = num_labels
        self.num_labels = num_labels
        config.output_hidden_states = True
        config.output_attentions = False
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, outputs_for_caching=False,**kwargs):
        last_hidden_state, pooled_output, hidden_states = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits,labels)
            return logits, hidden_states, loss
        else:
            if outputs_for_caching is True:
                return logits, hidden_states
            else:
                return logits
