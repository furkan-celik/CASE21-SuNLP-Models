import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from preprocessing import *
from embedding import *
from cnnmodel import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle

PARAMS = {"train_embed": True, "embedding": "NNLM", "dropout": 0.5, "batch_size": 16, "weights": 1} # Some hyperparameters to tune different runs

trainDfEng =  pd.read_json("./en-train.json", lines = True)

testData = pd.read_json("./test.json", lines = True)

trainDfEng.drop_duplicates(subset= "text",inplace= True)    # In original training data, there was a few redundant data, we choosed to drop those duplications to prevent data leakage

all_cleaned_texts = np.array([clean(text) for text in trainDfEng["text"]])  # Clean data in original form, you can use clean(text, True) to use stemmed versions

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_cleaned_texts)

all_encoded_texts = tokenizer.texts_to_sequences(all_cleaned_texts)
all_encoded_texts = np.array(all_encoded_texts)

maxlen = max([len(sent) for sent in all_encoded_texts]) # We choosed to take maxlength amongst sentences as our maxlen both in tokenizer and embedding
all_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(all_encoded_texts, maxlen= maxlen)

X_train, X_val, y_train, y_val = train_test_split(all_encoded_texts, trainDfEng["label"], test_size=0.2, random_state=42)

# Compact preprocessing of the test data
X_test = tf.keras.preprocessing.sequence.pad_sequences(np.array(tokenizer.texts_to_sequences(np.array([clean(text) for text in testData["text"]]))), maxlen= maxlen)

# Best working embedding was NNLM in our tests at subtask 2. We also calculate tokenizer and embedding matrix from all of the training data, instead of training split.
embedding_matrix = getEmbedding(all_cleaned_texts, tokenizer, PARAMS["embedding"])

model = get_model(embedding_matrix, maxlen, PARAMS)

model.fit(X_train, y_train, validation_data= (X_val, y_val), batch_size= PARAMS["batch_size"], epochs= 100, callbacks= [
    tf.keras.callbacks.EarlyStopping(patience= 3, verbose = 1, monitor = "val_auc", mode= "max"),   # We use AUC as our monitor instead of loss, since it follows trends in F1 Macro more closely in our tests
    tf.keras.callbacks.ModelCheckpoint(monitor = "val_auc", filepath = "./modelcnn", save_best_only = True, save_weights_only = True)
])

model.load_weights("./modelcnn")

print("Results in validation set: ")
print(classification_report(y_val, model.predict(X_val) > 0.5, digits= 4))

fpr, tpr, thresholds = metrics.roc_curve(y_val, model.predict(X_val))

y_pred = model.predict(X_val)

# To see how confident our model gets about it's predictions.
bestT = 0
bestF1 = 0
for t in thresholds:
  f1 = f1_score(y_val, y_pred > t)
  if f1 > bestF1:
    bestF1 = f1
    bestT = t

print("Best threshold: ", bestT)
print(classification_report(y_val, model.predict(X_val) > bestT))

y_pred_final = model.predict_proba(X_test)

with open("./ModelResults/" + "CNN-NNLM-TRAINED" + ".pkl", "wb") as fout:
  pickle.dump(y_pred, fout)