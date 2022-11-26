import torch
from torch import nn
import json

with open('config.json', 'rb') as f:
    CONFIG = json.loads(f.read())

CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def inference(tokenizer, discourse_type, discourse_text, essay_text, model):
    """
    Method to generate predictions on test samples, used in predict_deploy.py
    Takes user input of discourse_type, discourse_text, essay_text along with the tokenizer and finetuned model.
    :return: torch.tensor of probabilities for all 3 classes.
    """
    model.eval()
    input_discourse = discourse_type + ' ' + tokenizer.sep_token + ' ' + discourse_text
    tokenized_discourse = tokenizer.encode_plus(
        input_discourse,
        return_token_type_ids=False,
        return_attention_mask=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    tokenized_essay = tokenizer.encode_plus(
                            essay_text,
                            return_token_type_ids=False,
                            return_attention_mask=True,
                            max_length=512,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            return_tensors='pt',
                        )
    discourse_input_ids = tokenized_discourse['input_ids'].to(CONFIG['device'], non_blocking=True)
    discourse_attention_mask = tokenized_discourse['attention_mask'].to(CONFIG['device'], non_blocking=True)
    essay_input_ids = tokenized_essay['input_ids'].to(CONFIG['device'], non_blocking=True)
    essay_attention_mask = tokenized_essay['attention_mask'].to(CONFIG['device'], non_blocking=True)
    with torch.cuda.amp.autocast():
        logits = model(discourse_input_ids, discourse_attention_mask, essay_input_ids, essay_attention_mask)
        probs = nn.Softmax(dim=1)(logits)
    return probs