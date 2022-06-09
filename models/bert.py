import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention
from models.crf import CRF

@dataclass
class TokenClassifierOutput(object):
    """
    Base class for outputs of token classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    event_multi_logits: Optional[torch.FloatTensor] = None # 事件多标签
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predictions: Optional[torch.LongTensor] = None



class BertForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        

        return TokenClassifierOutput(
            logits=logits
        )

class BertCrfWithContraintForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        b_tags = list(range(0, self.num_labels-1, 2))
        i_tags = list(range(1, self.num_labels-1, 2))
        self.crf = CRF(
            self.num_labels, 
            batch_first=True,
            **{'b_tags': b_tags, 'i_tags': i_tags, 'o_tag': self.num_labels-1}
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        attention_mask = attention_mask == 1 # 转化成 bool 类型
        log_likelihood = self.crf(logits, labels, attention_mask, reduction='mean')

        if is_predict:
            predictions = self.crf.decode(logits)   # list[list[int]]
            predictions = torch.LongTensor(predictions) # no need to put it on GPU
            return TokenClassifierOutput(
                logits=logits,
                loss=-log_likelihood,
                predictions=predictions
            )
        
        return TokenClassifierOutput(
            logits=logits,
            loss=-log_likelihood
        )

class BertCrfForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        attention_mask = attention_mask == 1 # 转化成 bool 类型
        log_likelihood = self.crf(logits, labels, attention_mask, reduction='mean')

        if is_predict:
            predictions = self.crf.decode(logits)   # list[list[int]]
            predictions = torch.LongTensor(predictions) # no need to put it on GPU
            return TokenClassifierOutput(
                logits=logits,
                loss=-log_likelihood,
                predictions=predictions
            )
        
        return TokenClassifierOutput(
            logits=logits,
            loss=-log_likelihood
        )
        

class TokenPairBertForDD(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, max_trigger_len=4):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.max_trigger_span_len = max_trigger_len

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activate = F.gelu
        # self.border_attention = BertSelfAttention(config)
        # self.border_classifier = nn.Linear(config.hidden_size, 4)
        # self.trigger_attention = BertSelfAttention(config)
        # self.linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.trigger_classifier = nn.Linear(config.hidden_size, self.num_labels)
        #self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.padding = nn.Parameter(torch.Tensor(1, config.hidden_size))
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
       
        token_classifier_output = TokenClassifierOutput()
        sequence_output = outputs[0]
        sequence_output = self.activate(self.dropout(sequence_output))
        
        token_pair_output = sequence_output.clone()
        accumulate = sequence_output.clone()
        for i in range(self.max_trigger_span_len - 1):
            padding = self.padding.expand(input_ids.size(0), i+1, sequence_output.size(-1))
            
            accumulate += torch.cat([sequence_output[:, i+1:, :], padding], dim=-2)
            token_pair_output = torch.cat([token_pair_output, accumulate/(i+2)], dim=-2)


        token_pair_logits = self.trigger_classifier(token_pair_output)
        
        token_classifier_output.logits = token_pair_logits

        if is_predict:
            predictions = []
            batch_preds = torch.argmax(token_pair_logits, dim=-1)
            for preds in batch_preds:
                pred_for_one = [self.num_labels-1] * input_ids.size(-1)
                for i, p in enumerate(preds):
                    if p != self.num_labels - 1:
                        n = i // input_ids.size(-1)
                        left = i - input_ids.size(-1) * n
                        pred_for_one[left] = p * 2
                        for index in range(left+1, left+n+1):
                            if index < input_ids.size(-1):
                                pred_for_one[index] = p * 2 + 1
                predictions.append(pred_for_one)
            predictions = torch.LongTensor(predictions)
            token_classifier_output.predictions = predictions
        
        return token_classifier_output