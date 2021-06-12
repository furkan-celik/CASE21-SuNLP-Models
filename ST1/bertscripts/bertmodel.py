import torch
import transformers
import pytorch_lightning as pl
import torchmetrics
import pandas as pd

class EventTagger(pl.LightningModule):
  def __init__(self, n_classes: int, steps_per_epoch = None, n_epochs = None, learning_rate = 5e-5, BERT_MODEL_NAME = "roberta_base"):
    super().__init__()

    self.bert = transformers.AutoModel.from_pretrained(BERT_MODEL_NAME, return_dict = True)
    self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
    self.steps_per_epoch = steps_per_epoch
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.criterion = torch.nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels = None):
    output = self.bert(input_ids, attention_mask = attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)

    loss = 0
    if labels != None:
      loss = self.criterion(output, labels)

    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    loss, output = self(input_ids, attention_mask, labels)

    self.log("train_loss", loss, prog_bar = True, logger = True)

    return {"loss": loss, "predictions": output, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    
    loss, output = self(input_ids, attention_mask, labels)

    self.log("val_loss", loss, prog_bar = True, logger = True)

    return {"loss": loss, "predictions": output, "labels": labels}

  def test_step(self, batch, batch_idx):    # This does not expect labels since test dataset doesn't have public labels
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    loss, output = self(input_ids, attention_mask)

    # self.log("test_loss", loss, prog_bar = True, logger = True)

    return loss

  def training_epoch_end(self, outputs):
    labels = []
    predictions = []

    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels.int())

      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels)
    predictions = torch.stack(predictions)

    for i, name in enumerate(["label"]):    # During development we used both f1 and auc but auc gave problems due to the data distribution.
      #roc_score = torchmetrics.functional.auroc(predictions[:, i], labels[:, i])
      f1_macro = torchmetrics.functional.f1(predictions[:, i], labels[:, i], average= "macro", num_classes= 1)

      #self.logger.experiment.log_metric(f"{name}_roc_auc/Train", roc_score, self.current_epoch)
      self.logger.experiment.log_metric(f"{name}_f1macro/Train", f1_macro, self.current_epoch)

  def validation_epoch_end(self, outputs):
    labels = []
    predictions = []

    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels.int())

      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels)
    predictions = torch.stack(predictions)

    for i, name in enumerate(["label"]):
      #roc_score = torchmetrics.functional.auroc(predictions[:, i], labels[:, i])
      f1_macro = torchmetrics.functional.f1(predictions[:, i], labels[:, i], average= "macro", num_classes= 1)

      #self.logger.experiment.log_metric(f"{name}_roc_auc/Test", roc_score, self.current_epoch)
      self.logger.experiment.log_metric(f"{name}_f1macro/Test", f1_macro, self.current_epoch)

  def configure_optimizers(self):   # Scheduler can be changed with a one without hard_restarts
    optimizer = transformers.AdamW(self.parameters(), lr= self.learning_rate)

    warmup_steps = self.steps_per_epoch // 3    # First third of the epoch is warmup to fasten the training process
    total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

    # We use default 1 hard restart
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    return [optimizer], [scheduler]