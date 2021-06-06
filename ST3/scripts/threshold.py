from utils import *
from model import *
from evaluation import *
import argparse
import json
from torch.utils.data import TensorDataset
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--c', required=True)
    args = parser.parse_args()
    
    data_parameters = json.load(open(args.c+"/data_parameters.json","r"))
    model_parameters = json.load(open(args.c+"/model_parameters.json","r"))

    MODEL_SEED = model_parameters["MODEL_SEED"]
    DATA_SEED = data_parameters["DATA_SEED"]

    seed_everything(DATA_SEED)

    [TR_II,TR_AM,TR_LABEL,DE_II,DE_AM,DE_LABEL,TE_II,TE_AM,TE_LABEL],[train_codes,dev_codes,test_codes] = gimme_datasets(**data_parameters)

    seed_everything(MODEL_SEED)

    
    model = CorefClassifier.load_from_checkpoint("../models/bert-base-uncased-epoch=03-val_conll=0.9089.ckpt",
                        MODEL   = model_parameters["MODEL"],
                        TRAIN_DATA  = TensorDataset(TR_II,TR_AM,TR_LABEL),TRAIN_CODES = train_codes,
                        DEV_DATA    = TensorDataset(DE_II,DE_AM,DE_LABEL),DEV_CODES = dev_codes,
                        #TEST_DATA   = TensorDataset(DE_II,DE_AM,DE_LABEL),TEST_CODES = dev_codes,
                        TEST_DATA   = TensorDataset(TE_II,TE_AM,TE_LABEL),TEST_CODES = test_codes,
                        HIDDEN_UNIT1= model_parameters["HIDDEN_UNIT1"],
                        BATCH_SIZE  = model_parameters["BATCH_SIZE"],
                        LR          = model_parameters["LEARNING_RATE"],
                        EPS         = model_parameters["EPS"],
                        EPOCHS      = model_parameters["EPOCHS"],
                        FREEZE_BERT = model_parameters["FREEZE_BERT"])
    

    
    trainer = Trainer(gpus = 1)
    trainer.test(model)

    with open("../tensors/OLC.pkl","rb") as file:
        outputs,labels,test_codes = pickle.load(file)

    list_labels = labels.flatten().tolist()
    list_outputs= outputs.flatten().tolist()
    
    CSV_RESULTS= [test_codes[i]+[list_labels[i],list_outputs[i]] for i in range(len(test_codes))]

    import pandas as pd

    pd.DataFrame(CSV_RESULTS).to_csv("../CSVs/CSV_FOR_ANALYSIS.csv")

    THRESHOLDS = [0.3,0.4,0.5,0.6,0.65,0.7,0.8]
    BEST_CONLL = 0
    for THRESHOLD in THRESHOLDS:
        coref_metrics = evaluate(test_codes,outputs,labels,
                                 THRESHOLD,"test")
        print(coref_metrics)
    #     if coref_metrics["test_conll"] > BEST_CONLL:
    #         coref_metrics["THRESHOLD"] = THRESHOLD
    #         BEST_METRICS = coref_metrics
    # print(BEST_METRICS)

