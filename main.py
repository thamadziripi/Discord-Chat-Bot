# package imports
import json
import numpy as np 
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# Loading intent.json file
with open('Intent.json') as f:
    data = json.load(f)


training_sentences = []     # holds our training data
training_labels =[]     # holds our target labels
labels = []
responses = []


for intent in data['intents']:
    for texts in intent['text']:
        training_sentences.append(texts)
        training_labels.append(intent['intent'])
    responses.append(intent['responses'])

    if intent['intent'] not in labels:
        labels.append(intent['intent'])


num_classes = len(labels)


lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels =lbl_encoder.transform(training_labels)




