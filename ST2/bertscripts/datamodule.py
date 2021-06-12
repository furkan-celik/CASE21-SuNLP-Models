import torch
import transformers
import pytorch_lightning as pl
import pandas as pd

from datasetmodule import *

class Case21DataModule(pl.LightningDataModule):
  def __init__(self, train_df, val_df, test_df, tokenizer, batch_size = 8, max_token_len = 128):
    super().__init__()

    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_token_len = max_token_len

  def setup(self):
    self.train_dataset = Case21Dataset(
        self.train_df,
        self.tokenizer,
        self.max_token_len
    )

    self.val_dataset = Case21Dataset(
        self.val_df,
        self.tokenizer,
        self.max_token_len
    )

    self.test_dataset = Case21Dataset(
        self.test_df,
        self.tokenizer,
        self.max_token_len,
        testData = True
    )

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
        self.train_dataset,
        self.batch_size,
        shuffle = True,
        num_workers = 2
    )

  # Using 1 as batch_size made it easier to process our predictions with sklearn's functions. Since data is not huge, time complexity is not very much.
  def val_dataloader(self):
    return torch.utils.data.DataLoader(
        self.val_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 2
    )

  def test_dataloader(self):
    return torch.utils.data.DataLoader(
        self.test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 2
    )