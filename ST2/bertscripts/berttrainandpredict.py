import torch
import torchvision

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import transformers
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import  ModelCheckpoint

from sklearn.metrics import f1_score
import pickle

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_json("./en-train.json", lines= True)

df.drop_duplicates(subset= "sentence",inplace= True)

test_df = pd.read_json("./test.json", lines = True)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

BERT_MODEL_NAME = "roberta-base"  # roberta-base  bert-base-uncased bert-base-cased ablert-base-v2
tokenizer = transformers.AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

N_EPOCHS = 30
BATCH_SIZE = 32

data_module = Case21DataModule(train_df, val_df, test_df, tokenizer, BATCH_SIZE)
data_module.setup()

model = EventTagger(n_classes= 1, steps_per_epoch= len(train_df) // BATCH_SIZE, n_epochs= N_EPOCHS)

checkpoint_callback = ModelCheckpoint(monitor='val_loss')

trainer = pl.Trainer(max_epochs= N_EPOCHS, gpus = 1, progress_bar_refresh_rate= 30,
                     callbacks=[
                                EarlyStopping(monitor='val_loss', patience=2), 
                                checkpoint_callback,
                                ]
                     )

lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), data_module.val_dataloader())

# Plot with
fig = lr_finder.plot(suggest=True)
# fig.show()    # Uncomment to see learning rate graph

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

print("Learning rate set: ", new_lr)

model.learning_rate = new_lr

# Learning rate calculation can sometimes affect trainer process, so we reset our trainer
trainer = pl.Trainer(max_epochs= N_EPOCHS, gpus = 1, progress_bar_refresh_rate= 30,
                     callbacks=[
                                EarlyStopping(monitor='val_loss', patience=2), 
                                checkpoint_callback,
                                ]
                     )

trainer.fit(model, data_module)

model.freeze()

predictions = []
targets = []
for testObj in data_module.val_dataloader():
  _, test_pred = model(testObj["input_ids"], testObj["attention_mask"])
  targets.append(testObj["labels"])
  predictions.append(test_pred)

prediction_np = []
for pred in predictions:
  prediction_np.append(pred.cpu().flatten().numpy()[0])

targets_np = []
for t in targets:
  targets_np.append(int(t.flatten().numpy()[0]))

maxScore = float("-inf")
maxScoreThreshold = 0
for t in range(1, 99, 1):
  score = f1_score(np.array(targets_np), np.array(prediction_np) > (t / 100), average= "macro")

  if score > maxScore:
    maxScore = score
    maxScoreThreshold = t

print(f"For the threshold {maxScoreThreshold}, the system achievend highest F1-Macro score of {maxScore}")

predictions = []
for testObj in data_module.test_dataloader():
  _, test_pred = model(testObj["input_ids"], testObj["attention_mask"])
  predictions.append(test_pred)

prediction_np = []
for pred in predictions:
  prediction_np.append(pred.cpu().flatten().numpy()[0])

with open("./" + BERT_MODEL_NAME + "_optim.pkl", "wb") as fout:
  pickle.dump(prediction_np, fout)