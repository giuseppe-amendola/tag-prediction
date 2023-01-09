"""
API tag prediction
Input: Json with title and body of question
Endpoint (also known as route):
- /prediction: return tags predicted for the question submitted
"""

# Imports
import pickle
import numpy as np
import uvicorn
import nltk
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import normalize_corpus
import preprocessing

#packages for preprocessing
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')
#tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

class stackoverflow_question(BaseModel):
    """
    Question to predict must follow this format
    """
    Title: str
    Body: str

app = FastAPI() #initialize the app in fastAPI

# Loading the pipeline sklearn (TF-IDF + MultiOutputClassifier with Logistic Regression)
with open("model_new.pkl", "rb") as f:
    model = pickle.load(f)

# Load classes (coming from MultiLabelBinarizer)
with open("classes_new.pkl", "rb") as f:
    classes = pickle.load(f)

@app.get('/')
def index():
    return {'Tags prediction from question'}

@app.post('/prediction', summary="Return a list of predicted tags")
def get_prediction(data: stackoverflow_question):
    """
    Endpoint for predicting list of tags
    Arguments:
        data (stackoverflow_question): Title and body of StackOverflow question
    Returns:
        JSON: tags predicted
    """
    received = data.dict()

    concat_inputs = received["Title"] + " " + received["Body"]

    normalized_inputs = normalize_corpus(concat_inputs)

    pred_tags = model.predict([normalized_inputs])   

    zip_tags = zip(
        classes, 
        pred_tags[0])

    predicted_tags = dict(zip_tags)

    list_predicted = [k for k,v in predicted_tags.items() if v == 1]

    predicted = { 'predicted_tags' : list_predicted}

    return predicted


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000)