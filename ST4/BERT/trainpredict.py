from ST4.BERT.dataread import read_files
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import transformers
from NERDA.models import NERDA
from tqdm.notebook import tqdm

from dataread import *

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_sents, test_sents = train_test_split(read_files("./en-train.txt"), test_size = 0.2, random_state = 42)    # Read and split training data

training = {"sentences": [[w for w, l in sent] for sent in train_sents], "tags": [[l for w, l in sent] for sent in train_sents]}    # Split data into sentence and tags for training
validation = {"sentences": [[w for w, l in sent] for sent in test_sents], "tags": [[l for w, l in sent] for sent in test_sents]}

testing = {"sentences": [[w for w in sent] for sent in read_files("./test.txt")]}

# Tags our task uses
tag_scheme = ['B-trigger',
              'I-trigger',
              'B-participant',
              'I-participant',
              'B-organizer',
              'B-target',
              'I-target',
              'B-place',
              'B-etime',
              'I-etime',
              'B-fname',
              'I-fname',
              'I-organizer',
              'I-place']

transformer = "roberta-base"  # bert-base-uncased bert-base-cased albert-base-v2  roberta-base

max_len = max([len(sent) for sent in training["sentences"]])

model = NERDA(
    dataset_training = training,
    dataset_validation = validation,
    tag_scheme = tag_scheme,
    tag_outside = "O",
    transformer = transformer,
    max_len = max_len,
    dropout = 0.1,  #   Higher dropouts can be used.
    validation_batch_size = 8,
    hyperparameters = {"train_batch_size": 8}   # Higher batch sizes caused issues in our training runs, but we think that it is not ideal to leave this as low as 8.
)

model.train()

print(model.evaluate_performance(validation, batch_size = 1))

# Predict the test data and dump it to a text file. However, this may create minor format issues which we solved with our hands as post processing.
res = ""
for i in tqdm(range(len(testing["sentences"]))):
  res += testing["sentences"][i] + "\t" + model.predict([testing["sentences"][i]]) + "\n"

with open("./" + transformer + ".txt", "wb") as fout:
  fout.write(res)