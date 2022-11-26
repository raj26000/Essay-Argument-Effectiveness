import torch
from torch import nn
from transformers import AutoModel
import json

with open('config.json', 'rb') as f:
    CONFIG = json.loads(f.read())

CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MeanPoolingLayer(nn.Module):
    """
    Pooling layer to average out the hidden state vectors of all tokens in a sequence.
    Reduces the last_hidden_state model output from (batch_size, sequence_length, hidden_state_dim) to (batch_size, hidden_state_dim) by doing so.
    To avoid averaging vectors of special tokens, dot product is performed with attention mask by expanding it from 0/1 to a vector of zeros or ones for each token in each sequence.
    """
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        mask_sum = expanded_mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-9)
        masked_hidden_state = torch.sum(last_hidden_state * expanded_mask, 1)
        return masked_hidden_state / mask_sum


class DiscourseEffectivenessModel(nn.Module):
    """
    Custom model with Siamese architecture - two separate pretrained model + pooling heads.
    One head for discourse (discourse_type + discourse_text) and one for essay_text.
    The pooled outputs from both heads and their difference is concatenated before passing to dense layer.
    This is a better alternative to concatenating all texts into a single sentence and passing to a single BERT head.
    """
    def __init__(self, num_classes=3):
        super(DiscourseEffectivenessModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained_layer = AutoModel.from_pretrained(CONFIG['pretrained_model'])
        self.pooler = MeanPoolingLayer()
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(3 * self.pretrained_layer.config.hidden_size, num_classes)

    def forward(self, discourse_input_ids, discourse_attention_mask, essay_input_ids, essay_attention_mask):
        # Sentence A - BERT + mean pooling
        discourse_out = self.pretrained_layer(input_ids=discourse_input_ids, attention_mask=discourse_attention_mask)
        discourse_emb = self.pooler(discourse_out.last_hidden_state, discourse_attention_mask)
        # Sentence B - BERT + mean pooling
        essay_out = self.pretrained_layer(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
        essay_emb = self.pooler(essay_out.last_hidden_state, essay_attention_mask)
        # concat
        concat_emb = torch.cat([discourse_emb, essay_emb, torch.abs(essay_emb - discourse_emb)], dim=-1)
        x = self.dropout(concat_emb)
        x = self.fc(x)
        return x