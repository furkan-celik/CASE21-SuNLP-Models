# CASE21-SuNLP-Models for Shared Task 1 of the Case21 Competition

This repository contains our group’s efforts in the multilingual protest news detection shared task, which is organized as a part of the Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE) Workshop. Task 1 of the shared task is about detection and classification of socio-political and crisis event information in document, sentence, cross-event, and token levels. 

- Document and sentence level detections are binary classification of sentences by having past or ongoing event reference. (Subtask 1 & 2 relatively)
- Cross-event level is clustering sentences by their event mention and evaluate whether two sentences mention the same event or not. (Subtask 3)
- Token level is the detection of a set of arguments each trigger has alongside the detection of the trigger itself. (Subtask 4)

The shared task contained cross language datasets; namely, those are English, Spanish and Portuguese. Among them, due to the time limitations, we participated in all four subtasks only in the English language. 

We achieved good Macro F1 scores in both document and sentence level classification by using the RoBERTa model. Alongside that, the LexStem CNN model has two channels one for lexical/original form of the data and another one for the stemmed version of that sentence. LexStem consistently outperformed CNN and RNN based models both in validation and test datasets. Alongside those models, we have used the weighted soft voting ensemble technique with the grid search approach to increase the F1 score of the RoBERTa model for not confident decisions of RoBERTa (between %10 and %90). This ensemble model qualified as 3rd place in the sentence classification task at the public leaderboard alongside RoBERTa achieving 3rd place in the document level classification.

Similarly in the token level event extraction task, our transformer-LSTM-CRF architecture outperforms regular transformers significantly. Like document and sentence levels, RoBERTa has proven to be the best performer model in this context. Using LSTM-CRF architecture on top of a transformer consistently outperformed transformer models in our tests with both the validation and test data. Due to a minor submission format, our efforts had not represented accurately in the public leaderboard but we would achieve 2nd place in the public dataset with our best model of RoBERTa-LSTM-CRF.

Model  | Validation Macro F1 | Test Macro F1
| :------------ |:---------------:| -----:|
LSTM  | 0.82  | 0.68
GRU  | 0.83  | 0.64
CNN-LexStem  | 0.88  | 0.71
RoBERTa  | 0.88  | 0.82
RoBERTa+RNN  | 0.89  | 0.83
RoBERTa+LexStem  | 0.88  | 0.84
Table 1: Sentence level classification results (Subtask 2)


Model  | Validation Macro F1 | Test Macro F1
| :------------ |:---------------:| -----:|
LSTM  | 0.82  | 0.77
CNN-LexStem  | 0.82  | 0.78
BERT  | 0.84  | 0.80
Albert  | 0.84  | 0.81
RoBERTa  | 0.86  | 0.81
Table 2: Document level classification results (Subtask 1)


Model | Data | Validation CONLL Score | Test CONLL Score
| :------------ |:--------------- |:---------------:| -----:|
BERT      | RAW  | 77.70  | 74.83
Ensemble  | RAW  | 79.01  | 74.27
Bert      | EXT  | 80.54  | 78.45
Ensemble  | EXT  | 80.03  | 78.66
Table 3: Cross-event classification results (Subtask 3)


Model  | Validation Macro F1 | Test Macro F1
| :------------ |:---------------:| -----:|
BERT     | 0.70  | 0.69
RoBERTa  | 0.72  | 0.74
BERT-BiLSTM-CRF     | 0.76  | 0.75
RoBERTa-BiLSTM-CRF  | 0.76  | 0.76
Table 4: Document level classification results (Subtask 4)

This repository contains both scripts and notebooks necessary to run each models for each subtask. To reproduce our works, one can use hyperparameters given in the Appendix.

# Appendix

In this part, we would like to give specific details of our experiments for reproducing purposes.

## Subtask 1 \& 2  CNN Model

- Dropout: 0.5
- Convolution 1d: 128 filter, kernel size 3x3, ReLU activation function, same padding
- Dense Layer: 64 filter, ReLU activation 
- Dense Output Layer: Sigmoid; Kernel Regulazer: 1e-4 L2 Rate
- Loss Function: binary cross entropy
- Optimizer: Adam 
- Learning Rate: 1e-3
- Fine Tune Embeddings: true, 
- Embedding Mode: NNLM
- Batch Size: 16
- Earlystopping Patience: 3
- Earlystopping Monitor: val\_auc

## Subtask 1 \& 2  BERT Model

- Number of Epochs: 30
- Batch Size: 32
- Learning Rate (ST1): 2e-4
- Learning Rate (ST2): 5e-5
- Optimizer: AdamW

### Scheduler

- Scheduler: cosine\_with\_hard\_restarts\_schedule\_with\_warmup
- Steps per Epoch: 232
- Warm Up Steps: 77
- Total Steps: 6883
- Number of Cycles: 1

## Subtask 2 - LexStem

- Dropout: 0.5
- Number of Channels: 2
- Channel 1 Convolution 1d: 128 filter, kernel size 3x3, ReLU activation function, same padding
- Channel 2 Convolution 1d: 128 filter, kernel size 3x3, ReLU activation function, same padding
- Dense Layer: 64 filter, ReLU activation 
- Dense Output Layer: Sigmoid; Kernel Regulazer: 1e-4 L2 rate 
- Loss Function: Binary Cross Entropy
- Optimizer: Adam 
- Learning Rate: 1e-3
- Fine Tune Embeddings: True, 
- Embedding Mode: NNLM
- Batch Size: 16
- Earlystopping Patience: 3
- Earlystopping Monitor: Validation AUC

## Subtask 2 - BiLSTM and BiGRU

- Dropout: 0.5
- Recurrent Dropout: 0
- Batch Size: 16
- Epoch Size: 100
- Loss Function: Binary Cross Entropy
- Optimizer: Adam 
- Learning Rate: 1e-3

## Subtask 3 - BERT & RoBERTa & ALBERT

- Model Seed: 77
- Data Seed: 22
- Batch Size: 32
- Learning Rate: 2.75e-05
- Eps: 1e-8
- Optimizer: AdamW

## Subtask 4 - BERT

- Max\_len: 256
- Dropout: 0.1
- Validation Batch Size: 8
- Train Batch Size: 8 
- Optimizer: AdamW

## Subtask 4 - LSTM-CRF

- Embedder Type: roberta-base 
- Batch Size: 16 
- Num of Epochs: 70
- Optimizer: AdamW
- Learning Rate: 2e-5
- L2 Rate: 1e-8
- Dropout: 0.5
- BiLSTM Input Size: 768
- BiLSTM Hidden Size: 200
- 1 LSTM Layer


