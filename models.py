import torch
from torch import nn
from transformers import AutoModel
import json

with open('config.json', 'rb') as f:
    CONFIG = json.loads(f.read())

CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MeanPoolingLayer(nn.Module):
    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        mask_sum = expanded_mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-9)
        masked_hidden_state = torch.sum(last_hidden_state * expanded_mask, 1)
        return masked_hidden_state / mask_sum

class DiscourseEffectivenessModel(nn.Module):
    def __init__(self, num_classes=3):
        super(DiscourseEffectivenessModel, self).__init__()
        self.num_classes = num_classes
        self.pretrained_layer = AutoModel.from_pretrained(CONFIG['pretrained_model'])
        self.pooler = MeanPoolingLayer()
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.pretrained_layer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.pretrained_layer(input_ids=input_ids, attention_mask=attention_mask)
        x = self.pooler(x.last_hidden_state, attention_mask)
        x = self.dropout(x)
        x = self.fc(x)
        return x

@torch.no_grad()
def inference(tokenizer, discourse_type, discourse_text, essay_text, model):
    model.eval()
    input_text = discourse_type + ' ' + tokenizer.sep_token + ' ' + discourse_text + ' ' + tokenizer.sep_token + ' ' + essay_text
    tokenized_data = tokenizer.encode_plus(
        input_text,
        return_token_type_ids=False,
        return_attention_mask=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = tokenized_data['input_ids'].to(CONFIG['device'], non_blocking=True)
    attention_mask = tokenized_data['attention_mask'].to(CONFIG['device'], non_blocking=True)
    with torch.cuda.amp.autocast():
        logits = model(input_ids, attention_mask)
        probs = nn.Softmax(dim=1)(logits)
    return probs