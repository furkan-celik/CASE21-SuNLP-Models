import torch
import transformers
import pandas as pd

class Case21Dataset(torch.utils.data.Dataset):
  def __init__(self, data: pd.DataFrame, tokenizer: transformers.BertTokenizer, max_token_len: int = 128, testData = False):
    
    self.data = data
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len
    self.test_data = testData

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.sentence
    labels = []

    if not self.test_data:
      labels = data_row["label"]

    encoding = self.tokenizer.encode_plus(
        comment_text,
        add_special_tokens = True,
        max_length = self.max_token_len,
        return_token_type_ids = False,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt",
    )

    if not self.test_data:
      return dict(
          comment_text = comment_text,
          input_ids = encoding["input_ids"].flatten(),
          attention_mask = encoding["attention_mask"].flatten(),
          labels = torch.FloatTensor(labels)
      )
    else:   # This is rquired since our test data does not have labels, we need to discard it.
      return dict(
          comment_text = comment_text,
          input_ids = encoding["input_ids"].flatten(),
          attention_mask = encoding["attention_mask"].flatten()
      )