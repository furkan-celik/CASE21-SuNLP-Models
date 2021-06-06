import pickle
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning
from pytorch_lightning import Trainer,seed_everything

from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
from transformers import AutoModel,AdamW,get_linear_schedule_with_warmup


from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall, F1
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from evaluation import *

class CorefClassifier(LightningModule):

    def __init__(self,MODEL,TRAIN_DATA,TRAIN_CODES,DEV_DATA,DEV_CODES,TEST_DATA,TEST_CODES,HIDDEN_UNIT1,BATCH_SIZE,LR,EPS,EPOCHS,FREEZE_BERT=False):

        super(CorefClassifier, self).__init__()  
        #self.save_hyperparameters()
        
        self.BEST_THRESHOLD = 0

        self.train_data  = TRAIN_DATA
        self.train_codes = TRAIN_CODES

        self.dev_data   = DEV_DATA
        self.dev_codes  = DEV_CODES

        self.test_data  = TEST_DATA
        self.test_codes = TEST_CODES

        self.model = AutoModel.from_pretrained(MODEL)
        self.hidden_unit1 = HIDDEN_UNIT1
        
        if self.hidden_unit1:
            self.hidden_layer1 = nn.Linear(768, self.hidden_unit1)
            self.hidden_layer2 = nn.Linear(self.hidden_unit1, 1)
        else:
            self.hidden_layer1 = nn.Linear(768, 1)

        self.lossfn = nn.BCELoss()
        self.batch_size = BATCH_SIZE
        self.lr  = LR
        self.eps = EPS
        self.epochs = EPOCHS
        
        if FREEZE_BERT:
            for param in self.model.parameters():
                param.requires_grad = False
        
        #Metrics
        self.valid_metrics = MetricCollection([Accuracy(),
                                               Precision(num_classes=1, average='macro'),
                                               Recall(num_classes=1, average='macro'),
                                               F1(num_classes=1, average='macro')
                                              ])
        
        self.test_metrics = MetricCollection([Accuracy(),
                                               Precision(num_classes=1, average='macro'),
                                               Recall(num_classes=1, average='macro'),
                                               F1(num_classes=1, average='macro')
                                              ])
        
    def forward(self, input_ids, attention_mask):
        
        X = self.model(input_ids=input_ids,attention_mask=attention_mask)
        X = X[0][:, 0, :] # Extract the last hidden state of the token `[CLS]` for classification task
        
        if self.hidden_unit1:
            X = self.hidden_layer1(X)
            X = self.hidden_layer2(X)
        else:
            X = self.hidden_layer1(X)
        o = torch.sigmoid(X)
        return o
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data,
                                      num_workers=5,
                                      batch_size=len(self.train_data) if self.batch_size == 1 else self.batch_size)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(self.dev_data,
                                    num_workers=5,
                                    batch_size=len(self.dev_data) if self.batch_size == 1 else self.batch_size)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data,
                                    num_workers=5,
                                    batch_size=len(self.test_data) if self.batch_size == 1 else self.batch_size)
        return test_dataloader
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids,attention_mask)
        loss = self.lossfn(outputs, labels)
        self.log('train_loss', loss,on_step=True, on_epoch=True)
        return loss
    
    #def training_epoch_end(self, training_step_outputs):
    #    print(type(training_step_outputs),len(training_step_outputs),type(training_step_outputs[0]),len(training_step_outputs[0]))

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids,attention_mask)
        loss = self.lossfn(outputs, labels)
        self.log('val_loss', loss, on_epoch=True)#, sync_dist=True)
        
        for metric in self.valid_metrics(outputs,labels.to(torch.int)):
            self.log("val_"+metric,self.valid_metrics[metric], on_epoch=True)#, sync_dist=True)                
        return outputs,labels
    
    def validation_epoch_end(self, validation_step_outputs):
        outputs = torch.cat([i[0] for i in validation_step_outputs]).detach()
        labels = torch.cat([i[1] for i in validation_step_outputs]).detach()

        # if self.BEST_THRESHOLD == 0:
        #     max_conll = 0
        #     THRESHOLDS = [0.4,0.5,0.55,0.6,0.65,0.7]
        #     for THRESHOLD in THRESHOLDS:
        #         t_m = evaluate(self.dev_codes,outputs,labels,THRESHOLD,"val")
        #         if t_m["val_conll"] >= max_conll:
        #             max_conll = t_m["val_conll"]
        #             self.BEST_THRESHOLD = THRESHOLD
    
        coref_metrics = evaluate(self.dev_codes,
                                 outputs,labels,
                                 0.5,"val")

        #self.log(self.BEST_THRESHOLD,"VAL_TRESHOLD")

        for metric in coref_metrics:
            self.log(metric,coref_metrics[metric])

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids,attention_mask)
        for metric in self.test_metrics(outputs,labels.to(torch.int)):
            self.log("test_"+metric,self.test_metrics[metric], on_epoch=True) 
        return outputs,labels

    def test_epoch_end(self,test_step_outputs):
        outputs = torch.cat([i[0] for i in test_step_outputs]).detach()
        labels = torch.cat([i[1] for i in test_step_outputs]).detach()

        with open("../tensors/OLC.pkl","wb") as file:
            pickle.dump([outputs,labels,self.test_codes],file)

        coref_metrics = evaluate(self.test_codes,outputs,labels,
                                 0.5,"test")

        for metric in coref_metrics:
            self.log(metric,coref_metrics[metric])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),lr = self.lr, eps = self.eps)
        #scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                            num_warmup_steps = 3,
        #                                            num_training_steps = round(len(self.train_data)/self.batch_size+0.49)*self.epochs)
        return optimizer#[optimizer],[scheduler]