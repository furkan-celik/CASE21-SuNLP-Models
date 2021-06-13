import torch
import torch.nn as nn
from transformers import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from singlechain_crf import LinearCRF
from typing import Tuple
from overrides import overrides

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD = "<PAD>"

context_models = {
    'bert-base-uncased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'bert-base-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'bert-large-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer}
}

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(self, label_size: int, input_dim:int,
                 hidden_dim: int,
                 drop_lstm:float=0.5,
                 num_lstm_layers: int =1):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = label_size
        print("[Model Info] Input size to LSTM: {}".format(input_dim))
        print("[Model Info] LSTM Hidden Size: {}".format(hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)
        self.hidden2tag = nn.Linear(hidden_dim, self.label_size)

    @overrides
    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tag(feature_out)
        return outputs[recover_idx]

class LinearEncoder(nn.Module):

    def __init__(self, label_size:int, input_dim:int):
        super(LinearEncoder, self).__init__()

        self.hidden2tag = nn.Linear(input_dim, label_size)

    @overrides
    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        outputs = self.hidden2tag(word_rep)
        return outputs

class TransformersEmbedder(nn.Module):
    """
    Encode the input with transformers model such as
    BERT, Roberta, and so on.
    """

    def __init__(self, transformer_model_name: str,
                 parallel_embedder: bool = False):
        super(TransformersEmbedder, self).__init__()
        output_hidden_states = False ## to use all hidden states or not
        print(colored(f"[Model Info] Loading pretrained language model {transformer_model_name}", "red"))

        self.model = context_models[transformer_model_name]["model"].from_pretrained(transformer_model_name,
                                                                                   output_hidden_states= output_hidden_states,
                                                                                     return_dict=False)
        self.parallel = parallel_embedder
        if parallel_embedder:
            self.model = nn.DataParallel(self.model)
        """
        use the following line if you want to freeze the model, 
        but don't forget also exclude the parameters in the optimizer
        """
        # self.model.requires_grad = False

    def get_output_dim(self):
        ## use differnet model may have different attribute
        ## for example, if you are using GPT, it should be self.model.config.n_embd
        ## Check out https://huggingface.co/transformers/model_doc/gpt.html
        ## But you can directly write it as 768 as well.
        return self.model.config.hidden_size if not self.parallel else self.model.module.config.hidden_size

    def forward(self, word_seq_tensor: torch.Tensor,
                       orig_to_token_index: torch.LongTensor, ## batch_size * max_seq_leng
                        input_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param word_seq_tensor: (batch_size x max_wordpiece_len x hidden_size)
        :param orig_to_token_index: (batch_size x max_sent_len x hidden_size)
        :param input_mask: (batch_size x max_wordpiece_len)
        :return:
        """
        word_rep, _ = self.model(**{"input_ids": word_seq_tensor, "attention_mask": input_mask})
        ##exclude the [CLS] and [SEP] token
        # _, _, word_rep = self.model(**{"input_ids": word_seq_tensor, "attention_mask": input_mask})
        # word_rep = torch.cat(word_rep[-4:], dim=2)
        batch_size, _, rep_size = word_rep.size()
        _, max_sent_len = orig_to_token_index.size()
        return torch.gather(word_rep[:, 1:, :], 1, orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))

class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
                                             parallel_embedder=config.parallel_embedder)
        if config.hidden_dim > 0:
            self.encoder = BiLSTMEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim(),
                                         hidden_dim=config.hidden_dim, drop_lstm=config.dropout)
        else:
            self.encoder = LinearEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim())
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)
        self.pad_idx = config.label2idx[PAD]


    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        lstm_scores = self.encoder(word_rep, word_seq_lens)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score - labeled_score

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        features = self.encoder(word_rep, word_seq_lens)
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return bestScores, decodeIdx