import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from models.bert import TokenClassifierOutput
from models.crf import CRF

class CustomedEmbedding(nn.Module):
    def __init__(self, embedding_file=None, vocab_size=21128, hidden_size=300, **kwargs):
        super().__init__()
        if not embedding_file:
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
        else:
            weight = torch.load(embedding_file)
            self.embeddings = nn.Embedding.from_pretrained(weight)
    
    def forward(self, inputs):
        return self.embeddings(inputs)

class BiLSTM(nn.Module):
    def __init__(self, num_labels, embedding_file=None, hidden_size=300, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        # 21128 is the vocabulary size of bert
        self.from_bert = kwargs.get('from_bert', False)
        self.embeddings = CustomedEmbedding(embedding_file, hidden_size=hidden_size)
        if self.from_bert:
            assert embedding_file is not None
            self.embed_transfer = nn.Linear(self.embeddings.embeddings.embedding_dim, hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, dropout=0.5,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size*2, self.num_labels)
        

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model = cls(**kwargs)
        if not model_path or not os.path.exists(model_path):
            logger.info(f'Initializing model.')
            return model
        model_path = os.path.join(model_path, 'pytorch_model.bin')
        logger.info(f'Loading model from {model_path}.')
        model.load_state_dict(torch.load(model_path))
        return model
    
    def forward(self, input_ids, attention_mask=None, labels=None, is_predict=False, **kwargs):
        embeddings = self.embeddings(input_ids)
        if self.from_bert:
            embeddings = self.embed_transfer(embeddings)
        output, (h_n, c_n) = self.bilstm(embeddings)
        logits = self.classifier(output)
        
        return TokenClassifierOutput(
            logits=logits
        )


class BiLSTMCRF(nn.Module):
    def __init__(self, num_labels, embedding_file=None, hidden_size=300, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        # 21128 is the vocabulary size of bert
        self.from_bert = kwargs.get('from_bert', False)
        self.embeddings = CustomedEmbedding(embedding_file, hidden_size=hidden_size)
        if self.from_bert:
            assert embedding_file is not None
            self.embed_transfer = nn.Linear(self.embeddings.embeddings.embedding_dim, hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, dropout=0.5,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size*2, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model = cls(**kwargs)
        model_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(model_path):
            logger.info(f'Initializing model.')
            return model
        logger.info(f'Loading model from {model_path}.')
        model.load_state_dict(torch.load(model_path))
        return model
    
    def forward(self, input_ids, attention_mask=None, labels=None, is_predict=False, **kwargs):
        embeddings = self.embeddings(input_ids)
        if self.from_bert:
            embeddings = self.embed_transfer(embeddings)
        output, (h_n, c_n) = self.bilstm(embeddings)
        logits = self.classifier(output)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.uint8)
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

class CustomedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding_size = (self.kernel_size - 1) >> 1
        self.CNN = nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=self.kernel_size,
                             stride=1,
                             padding=self.padding_size)
        
        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        preds = self.CNN(inputs)
        preds = self.activation(preds)
        return preds

class DMCNN(nn.Module):
    def __init__(self, num_labels, embedding_file=None, hidden_size=200, max_seq_len=256, pf_dim=5, **kwargs):
        super().__init__()
        self.from_bert = kwargs.get('from_bert', False)
        self.embeddings = CustomedEmbedding(embedding_file, hidden_size=hidden_size)
        self.embedding_size = 300 #self.embeddings.embeddings.embedding_dim
        if self.from_bert:
            assert embedding_file is not None
            self.embed_transfer = nn.Linear(self.embeddings.embeddings.embedding_dim, self.embedding_size)
        
        self.num_labels, self.seq_len = num_labels, max_seq_len
        self.pf_dim, self.hidden_size = pf_dim, hidden_size
        self.pf_embeddings = nn.Embedding(self.seq_len, self.pf_dim) # position feature
        self.cnn = CustomedCNN(self.embedding_size+self.pf_dim, self.hidden_size)
        self.dropout = nn.Dropout(0.5)
        # 3*self.embedding_size for lexical level feature
        # 2*hidden_size for sentence level feature
        self.fc = nn.Linear(3*self.embedding_size + 2*self.hidden_size,
                            self.num_labels, bias=True)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model = cls(**kwargs)
        model_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(model_path):
            logger.info(f'Initializing model.')
            return model
        logger.info(f'Loading model from {model_path}.')
        model.load_state_dict(torch.load(model_path))
        return model

    def forward(self, inputs, **kwargs):
        token_embeddings = self.embeddings(inputs) # (bs, sl, es) : es -> embedding size, bs -> batch size, sl -> sequence length
        if self.from_bert:
            token_embeddings = self.embed_transfer(token_embeddings)
        assert token_embeddings.size(1) == self.seq_len
        padding_preds = [0.0] * self.num_labels
        padding_preds[-1] = 1.0
        padding_preds = torch.FloatTensor([padding_preds]*inputs.size(0)).unsqueeze(1).to(inputs.device)
        seq_preds = padding_preds # (bs, 1, nc): nc -> the number of classes
        for i in range(1, self.seq_len-1):
            # lexical level feature
            llf = token_embeddings[:, i-1: i+2, :] 
            llf = llf.view(-1, 3 * self.embedding_size) # (bs, es * 3)
            # positional embeddings
            pos_index = list(range(i, 0, -1)) + list(range(0, self.seq_len-i))
            pos_index = torch.LongTensor([pos_index]*inputs.size(0)).to(inputs.device)
            pf_embeddings = self.pf_embeddings(pos_index) # (bs, sl, 5)
            # setence feature input
            sf_input = torch.cat((token_embeddings, pf_embeddings), dim=-1) # (bs, sl, 5 + es)
            sf_out = self.cnn(sf_input) # (bs, fn, sl): fn -> the number of filters
            sf_out = self.dynamic_pooling(sf_out, index=i) # (bs, 2 * fn)
            sf_out = self.dropout(sf_out)
            all_features = torch.cat((sf_out, llf), dim=-1) # (bs, 2 * fn + 3 * es)
            preds = self.fc(all_features) # (bs, nc)
            seq_preds = torch.cat((seq_preds, preds.unsqueeze(1)), dim=1)

        seq_preds = torch.cat((seq_preds, padding_preds), dim=1) # (bs, sl, cn)
        
        return TokenClassifierOutput(
            logits=seq_preds
        )

    def dynamic_pooling(self, inputs, index):
        # inputs: (bs, fn, sl)
        max_left = torch.max(inputs[:, :, :index], dim=-1, keepdim=False)[0] # (bs, fn)
        max_right = torch.max(inputs[:, :, index:], dim=-1, keepdim=False)[0] # (bs, fn)
        output = torch.cat((max_left, max_right), dim=-1) # (bs, 2 * fn)
        return output

if __name__ == '__main__':

    dmcnn = DMCNN(num_labels=10)
    inputs = torch.LongTensor(list(range(4*256))).view(4, 256)
    outputs = dmcnn(inputs)
