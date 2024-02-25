from fastapi import FastAPI, status, UploadFile
import uvicorn
import json
import torch
from transformers import AutoTokenizer, AutoModel
from models import DiscourseEffectivenessModel
from infer import inference

app = FastAPI()
with open('config.json', 'r') as f:
    config = json.load(f)

config['device'] = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

model = DiscourseEffectivenessModel().to(config['device'])
model.load_state_dict(torch.load(config['saved_model_checkpoint'], map_location=config['device']))


@app.post('/evaluateDiscourse', status_code=status.HTTP_200_OK)
def predict_discourse_effectiveness(essay_file: UploadFile, discourse_type: str, discourse_text: str):
    essay_text = essay_file.file.read()
    probs = inference(tokenizer, discourse_type, discourse_text, str(essay_text), model)
    print(probs)
    return {"discourse_type": discourse_type,
            "discourse_text": discourse_text,
            "essay_text": str(essay_text),
            "discourse_effectiveness_scores": {"Effective": round(probs[0][0].item(), 2),
                                               "Adequate": round(probs[0][1].item(), 2),
                                               "Ineffective": round(probs[0][2].item(), 2)
                                               }
            }


if __name__ == "__main__":
    uvicorn.run(app, host=config['app_host'], port=config['app_port'])
