import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from preprocessing import *
from embedding import *
from lexstemmodel import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle

PARAMS = {"train_embed": True, "embedding": "NNLM", "dropout": 0.5, "seq_dropout": 0, "batch_size": 16, "weights": 1} # Some hyperparameters to tune different runs

trainDfEng =  pd.read_json("./en-train.json", lines = True)

trainDfEng.drop_duplicates(subset= "sentence",inplace= True)    # In original training data, there was a few redundant data, we choosed to drop those duplications to prevent data leakage

testData = pd.read_json("./test.json", lines = True)

# Tokenize and preprocess data in lexical (original) form.
all_cleaned_texts = np.array([clean(text) for text in trainDfEng["sentence"]])  # Clean data in original form, you can use clean(text, True) to use stemmed versions

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_cleaned_texts)

all_encoded_texts = tokenizer.texts_to_sequences(all_cleaned_texts)
all_encoded_texts = np.array(all_encoded_texts)

maxlen = max([len(sent) for sent in all_encoded_texts]) # We choosed to take maxlength amongst sentences as our maxlen both in tokenizer and embedding
all_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(all_encoded_texts, maxlen= maxlen)

X_train, X_val, y_train, y_val = train_test_split(all_encoded_texts, trainDfEng["label"], test_size=0.2, random_state=42)

# Tokenize and preprocess data in stemmed form.
# We do not train tokenizer again for stemmed version and hence use the same embedding matrix at the model.
all_cleaned_texts_stemmed = np.array([clean(text, True) for text in trainDfEng["sentence"]])  # You can use clean(text) to use lexical versions

all_encoded_texts_stemmed = tokenizer.texts_to_sequences(all_cleaned_texts_stemmed)
all_encoded_texts_stemmed = np.array(all_encoded_texts_stemmed)

all_encoded_texts_stemmed = tf.keras.preprocessing.sequence.pad_sequences(all_encoded_texts_stemmed, maxlen= maxlen)

X_train_Stem, X_val_Stem, y_train_Stem, y_val_Stem = train_test_split(all_encoded_texts_stemmed, trainDfEng["label"], test_size=0.2, random_state=42)

# Compact preprocessing of the test data
X_test = tf.keras.preprocessing.sequence.pad_sequences(np.array(tokenizer.texts_to_sequences(np.array([clean(text) for text in testData["sentence"]]))), maxlen= maxlen)
X_test_Stem = tf.keras.preprocessing.sequence.pad_sequences(np.array(tokenizer.texts_to_sequences(np.array([clean(text, True) for text in testData["sentence"]]))), maxlen= maxlen)

# Best working embedding was NNLM in our tests at subtask 2. We also calculate tokenizer and embedding matrix from all of the training data, instead of training split.
embedding_matrix = getEmbedding(all_cleaned_texts, tokenizer, PARAMS["embedding"])

model = get_model(embedding_matrix, maxlen, PARAMS)

model.fit([X_train, X_train_Stem], y_train, validation_data= ([X_test, X_test_Stem], y_test), batch_size= 16, epochs= 100, callbacks= [
    tf.keras.callbacks.EarlyStopping(patience= 3, verbose = 1, monitor = "val_auc", mode= "max"), 
    tf.keras.callbacks.ModelCheckpoint(monitor = "val_auc", filepath = "./modellexstem", save_best_only = True, save_weights_only = True, mode= "max"),
    MacroF1Callback()
])

model.load_weights("./modellexstem")

print("Results in validation set: ")
print(classification_report(y_val, model.predict([X_val, X_val_Stem]) > 0.5, digits= 4))

fpr, tpr, thresholds = metrics.roc_curve(y_val, model.predict([X_val, X_val_Stem]))

y_pred = model.predict([X_val, X_val_Stem])

# To see how confident our model gets about it's predictions.
bestT = 0
bestF1 = 0
for t in thresholds:
  f1 = f1_score(y_val, y_pred > t)
  if f1 > bestF1:
    bestF1 = f1
    bestT = t

print("Best threshold: ", bestT)
print(classification_report(y_val, model.predict([X_val, X_val_Stem]) > bestT))

y_pred_final = model.predict_proba([X_test, X_test_Stem])

with open("./" + "LEXSTEM-" +  PARAMS["embedding"] + "-" + PARAMS["train_embed"] + ".pkl", "wb") as fout:
  pickle.dump(y_pred, fout)