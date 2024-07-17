# Classifying Effectiveness of Argumentative elements in Student essays.
A **Streamlit**-based Web App for evaluating effectiveness of seven distinct argument types in student essays. Continue reading to understand more about the problem, dive deep into the training process (notebook linked) and reproduce the web app locally for inference.

## Describing the Problem
Given an essay text file along with the type of argumentative element to classify and its corresponding text, predict if the argument was **Effective**, **Adequate** or **Ineffective** -- in order of decreasing quality. The following argumentative discourse elements are supported for classification:
- **Lead:**  an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
- **Position:** an opinion or conclusion on the main question
- **Claim:** a claim that supports the position
- **Counterclaim:** a claim that refutes another claim or gives an opposing reason to the position
- **Rebuttal:** a claim that refutes a counterclaim
- **Evidence:** ideas or examples that support claims, counterclaims, or rebuttals.
- **Concluding Statement:** a concluding statement that restates the claims.

Hence, we are effectively solving a multiclass classification problem with three targets. The dataset for the same can be accessed from [**Kaggle**](https://www.kaggle.com/competitions/feedback-prize-effectiveness). 

## Training the Model
- Model training was performed using a **Siamese** architecture with the **DeBERTa-v3-base** model in PyTorch. The final model used was trained over **7 epochs** with an Early Stopping at **Patience 3**, to account for overfitting (The training did indeed abort at 6 epochs!). The approach used to model the text data is as follows: 
```
Sentence A: discourse_input = discourse_type [SEP] discourse_text 
Sentence B: essay_input = essay_text
```
- So, instead of concatenating all text into a single sentence and passing to a single pretrained head, we create two separate text inputs - one 
for discourse and one for essay - and pass these independently through two BERT + Pooling heads. The output embeddings from both and their 
difference is what is eventually concatenated (not the original sentences!). 
- Further details about the training process, which also employed techniques such as Gradient Accumulation and Mixed Precision Training, can be obtained from my Kaggle 
[**notebook**](https://www.kaggle.com/code/raj26000/pytorch-feedback-deberta-base-training). The best model is automatically saved during the course of training and the .pt checkpoint file can be downloaded for inference. Place the downloaded file in the project directory and update its path in the config appropriately. 
Alternately, you may choose to use my trained model checkpoint, which can be downloaded from [**here**](https://huggingface.co/raj26000/deberta-base-siamese-feedback-arguments/blob/main/saved_model_state_deberta_siamese_3-stack.pt).

## Inference Using Docker Image for Project
- Access and pull the docker image with `docker pull rajdeep26000/essay-arg-eff:0.0.1`
- The image exposes port 8008 from the container you create, bind the port to a desired number on your local while creating and running the container:
  `docker run -d --name <CONTAINER_NAME> -p <LOCAL_PORT>:8008 rajdeep26000/essay-arg-eff:0.0.1`
- Now use the following cURL to access the app via Postman:
  	- Upload the essay file as a `.txt`.
	- Choose the type of discourse element to evaluate (from the seven above)
	- Enter the text of the discourse or argument, which is essentially part of the above essay.
  `curl --location 'http://localhost:8008/evaluateDiscourse' \
    --form 'essay_file=@"/C:/Users/rajdeep.agrawal/Documents/D72CB1C11673.txt"' \
    --form 'discourse_type="Claim"' \
    --form 'discourse_text="Seeking others opinion can be very helpful and beneficial."'
  `

## Setup for reproducing Inference Locally
- Clone the repo using `git clone https://github.com/raj26000/Essay-Argument-Effectiveness.git`
- Using the [`environment.yml`](environment.yml) file, create the conda environment as follows:
```
conda env create -f environment.yml
```
This will install all the required dependencies for running the app locally.
And then activate it with `conda activate env_name`. Remember, the first line of the yml file has the name of the environment that will be created. Feel free to change.
- The [`config.json`](config.json) file contains 2 parameters:
	1. `pretrained_model`: The model architecture that was finetuned. I've used **DeBERTa**, you may run the training with one of your choice and replace here accordingly. Ex: 	`bert-base-uncased`, `roberta-base`.
	2. `saved_model_checkpoint`: Local path of the .pt file containing the best model's weights, obtained from the training notebook or downloaded from the [**link**](https://huggingface.co/raj26000/deberta-base-siamese-feedback-arguments/blob/main/saved_model_state_deberta_siamese_3-stack.pt) above.
- Run the FastAPI app with `python app.py` and access the endpoint as described above.

