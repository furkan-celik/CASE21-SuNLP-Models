import torch
from typing import List, Tuple, Dict, Any
import random

from termcolor import colored
from data import Instance

def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore

def write_results(filename: str, insts: List[Instance]):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.words)):
            words = inst.ori_words
            output = inst.labels
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)