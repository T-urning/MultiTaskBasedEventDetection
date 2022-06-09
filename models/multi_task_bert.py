import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention
from models.crf import CRF
from models.bert import TokenClassifierOutput

  

class MultiTaskBertCrfForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, event_nums, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_events = event_nums
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activate = F.gelu
        self.crf = CRF(self.num_labels, batch_first=True)
        self.event_attention = BertAttention(config)
        self.event_multi_label_classifier = nn.Linear(config.hidden_size, self.num_events)
        self.trigger_attention = BertAttention(config)
        #self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.trigger_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_predict=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.size(), attention_mask.device)
        token_classifier_output = TokenClassifierOutput()
        sequence_output = outputs[0]
        sequence_output = self.activate(self.dropout(sequence_output))
        event_output = self.event_attention(sequence_output, attention_mask=extended_attention_mask)[0]
        event_output = torch.mean(event_output, dim=1) # (bs, hs)
        event_output = self.dropout(event_output)
        if not is_predict:
            event_multi_logits = self.event_multi_label_classifier(event_output) # (bs, num_events)
            token_classifier_output.event_multi_logits = event_multi_logits

        trigger_output = sequence_output + event_output.view([-1, 1, event_output.size(-1)])
        #trigger_output = self.activate(self.dropout(trigger_output))
        trigger_output = self.trigger_attention(trigger_output, attention_mask=extended_attention_mask)[0]
        trigger_output = self.dropout(trigger_output)
        trigger_logits = self.trigger_classifier(trigger_output)
        
        attention_mask = attention_mask == 1 # 转化成 bool 类型
        log_likelihood = self.crf(trigger_logits, labels, attention_mask, reduction='mean')

        token_classifier_output.logits = trigger_logits
        token_classifier_output.loss = -log_likelihood
        if is_predict:
            predictions = self.crf.decode(trigger_logits)   # list[list[int]]
            predictions = torch.LongTensor(predictions) # no need to put it on GPU
            token_classifier_output.predictions = predictions
        

        return token_classifier_output

    
class MultiTaskBertCrfWithConstraintForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, event_nums, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_events = event_nums
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activate = F.gelu
        b_tags = list(range(0, self.num_labels-1, 2))
        i_tags = list(range(1, self.num_labels-1, 2))
        self.crf = CRF(
            self.num_labels, 
            batch_first=True,
            **{'b_tags': b_tags, 'i_tags': i_tags, 'o_tag': self.num_labels-1}
        )
        self.event_attention = BertAttention(config)
        self.event_multi_label_classifier = nn.Linear(config.hidden_size, self.num_events)
        self.trigger_attention = BertAttention(config)
        #self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.trigger_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_predict=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.size(), attention_mask.device)
        token_classifier_output = TokenClassifierOutput()
        sequence_output = outputs[0]
        sequence_output = self.activate(self.dropout(sequence_output))
        event_output = self.event_attention(sequence_output, attention_mask=extended_attention_mask)[0]
        event_output = torch.mean(event_output, dim=1) # (bs, hs)
        event_output = self.dropout(event_output)
        if not is_predict:
            event_multi_logits = self.event_multi_label_classifier(event_output) # (bs, num_events)
            token_classifier_output.event_multi_logits = event_multi_logits

        trigger_output = sequence_output + event_output.view([-1, 1, event_output.size(-1)])
        #trigger_output = self.activate(self.dropout(trigger_output))
        trigger_output = self.trigger_attention(trigger_output, attention_mask=extended_attention_mask)[0]
        trigger_output = self.dropout(trigger_output)
        trigger_logits = self.trigger_classifier(trigger_output)
        
        attention_mask = attention_mask == 1 # 转化成 bool 类型
        log_likelihood = self.crf(trigger_logits, labels, attention_mask, reduction='mean')

        token_classifier_output.logits = trigger_logits
        token_classifier_output.loss = -log_likelihood
        if is_predict:
            predictions = self.crf.decode(trigger_logits)   # list[list[int]]
            predictions = torch.LongTensor(predictions) # no need to put it on GPU
            token_classifier_output.predictions = predictions
        

        return token_classifier_output    
    
    
class MultiTaskBertForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, event_nums, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_events = event_nums
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activate = F.gelu
        self.event_attention = BertAttention(config)
        self.event_multi_label_classifier = nn.Linear(config.hidden_size, self.num_events)
        self.trigger_attention = BertAttention(config)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.trigger_classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_predict=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        attention_mask = self.get_extended_attention_mask(attention_mask, attention_mask.size(), attention_mask.device)
        token_classifier_output = TokenClassifierOutput()
        sequence_output = outputs[0]
        sequence_output = self.activate(self.dropout(sequence_output))
        #logits = self.classifier(sequence_output)
        event_output = self.event_attention(sequence_output, attention_mask=attention_mask)[0]
        event_output = torch.mean(event_output, dim=1) # (bs, hs)
        event_output = self.dropout(event_output)
        if not is_predict:
            
            event_multi_logits = self.event_multi_label_classifier(event_output) # (bs, num_events)
            token_classifier_output.event_multi_logits = event_multi_logits

        trigger_output = self.linear(sequence_output + event_output.view([-1, 1, event_output.size(-1)]))
        trigger_output = self.activate(self.dropout(trigger_output))
        trigger_output = self.trigger_attention(trigger_output, attention_mask=attention_mask)[0]
        trigger_output = self.dropout(trigger_output)
        trigger_logits = self.trigger_classifier(trigger_output)

        token_classifier_output.logits = trigger_logits
        
        return token_classifier_output





