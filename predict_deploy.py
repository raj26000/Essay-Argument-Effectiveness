from models import DiscourseEffectivenessModel
from infer import inference
import streamlit as st
from transformers import AutoTokenizer
import json
import torch
from io import StringIO

with open('config.json', 'rb') as f:
    CONFIG = json.loads(f.read())
CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.title("Essay Argument Effectiveness Evaluation")


@st.cache
def fetch_tokenizer_model(CONFIG):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    model = DiscourseEffectivenessModel().to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['saved_model_checkpoint'], map_location=CONFIG['device']))
    return tokenizer, model


st.subheader('Upload Essay File (*.txt):')
essay_file = st.file_uploader(label='', type=['txt'])
essay_text = ''
if essay_file is not None:
    stringio = StringIO(essay_file.getvalue().decode("utf-8"))
    text = stringio.read()
    essay_text = ' '.join([x for x in text.split()])
# essay_text = st.text_area('Complete essay text:', placeholder='Enter Here...')
st.subheader('Type of Discourse Element to evaluate:')
discourse_type = st.selectbox('', ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement'))
st.subheader('Text corresponding to above selected discourse type from essay:')
discourse_text = st.text_area('', placeholder='Enter Here...')
submit = st.button("Evaluate!")
if submit:
    tokenizer, model = fetch_tokenizer_model(CONFIG)
    with st.spinner("Generating Results:"):
        probs = inference(tokenizer, discourse_type, discourse_text, essay_text, model)
    st.subheader("Effectiveness Results (with Probabilities)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Effective", round(probs[0][0].item(), 2))
    col2.metric("Adequate", round(probs[0][1].item(), 2))
    col3.metric("Ineffective", round(probs[0][2].item(), 2))
