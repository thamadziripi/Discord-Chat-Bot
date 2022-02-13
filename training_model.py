# package imports
import json
import numpy as np 
import tensorflow 
import pickle
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

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = '<00v>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Training our model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

model.save('chat_model')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)