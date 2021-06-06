from utils import *
from model import *
from evaluation import *
import argparse
import json
from torch.utils.data import TensorDataset
from pytorch_lightning.loggers.neptune import NeptuneLogger

def single_cluster_baseline(pairs,labels,set_):

    cc = (labels.flatten()== 1.0).nonzero(as_tuple=False).flatten()

    key = list(map(pairs.__getitem__,cc))
    response = [i for i in pairs if i[0].split("_")[0] == i[1].split("_")[0]]
    coref_metrics = return_metrics(key,response,set_+"_"+"baseline")

    with open(f"{set_}_baseline_metrics.json","w") as file:
        json.dump(coref_metrics,file)

if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--c', required=True)
    args = parser.parse_args()
    
    data_parameters = json.load(open(args.c+"/data_parameters.json","r"))
    model_parameters = json.load(open(args.c+"/model_parameters.json","r"))
    MODEL_SEED = model_parameters["MODEL_SEED"]
    DATA_SEED = data_parameters["DATA_SEED"]
    



    [TR_II,TR_AM,TR_LABEL,DE_II,DE_AM,DE_LABEL,TE_II,TE_AM,TE_LABEL],[train_codes,dev_codes,test_codes] = gimme_datasets(**data_parameters)
    single_cluster_baseline(dev_codes,DE_LABEL,"val")
    single_cluster_baseline(test_codes,TE_LABEL,"test")

    seed_everything(MODEL_SEED)
    NEPTUNE_API="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGQ0YWZhNjAtYWRmOS00ZWYyLWI4NGYtMWFhZmI1MDI2YTk3In0="
    checkpoint_callback = ModelCheckpoint(
    monitor='val_conll',
    dirpath='../models/',
    filename="n_"+model_parameters["MODEL"]+'-{epoch:02d}-{val_conll:.4f}',
    save_top_k=1,
    mode='max',
)
    early_stop_callback = EarlyStopping(
       monitor='val_conll',
       min_delta=0.00,
       patience=3,
       verbose=False,
       mode='max')
    
    model = CorefClassifier(MODEL   = model_parameters["MODEL"],
                        TRAIN_DATA  = TensorDataset(TR_II,TR_AM,TR_LABEL),TRAIN_CODES = train_codes,
                        DEV_DATA    = TensorDataset(DE_II,DE_AM,DE_LABEL),DEV_CODES = dev_codes,
                        TEST_DATA   = TensorDataset(TE_II,TE_AM,TE_LABEL),TEST_CODES = test_codes,
                        HIDDEN_UNIT1= model_parameters["HIDDEN_UNIT1"],
                        BATCH_SIZE  = model_parameters["BATCH_SIZE"],
                        LR          = model_parameters["LEARNING_RATE"],
                        EPS         = model_parameters["EPS"],
                        EPOCHS      = model_parameters["EPOCHS"],
                        FREEZE_BERT = model_parameters["FREEZE_BERT"])
    
    neptune_logger = NeptuneLogger(api_key=NEPTUNE_API,
                               project_name="fatihbeyhan/CASE21-SUBTASK3",
                               params = {**model_parameters,**data_parameters})
    
    trainer = Trainer(max_epochs = model_parameters["EPOCHS"],
                        gpus = 1,
                        auto_lr_find=True,
                        auto_scale_batch_size='binsearch',
                        #gradient_clip_val= GRADIENT_CLIP,
                        #limit_train_batches = 1,
                        #limit_val_batches = 2,
                        #limit_test_batches = 1,
                        logger=neptune_logger,
                        #accelerator='ddp',
                        callbacks = [checkpoint_callback,early_stop_callback]
                 )
    trainer.tune(model)
    trainer.fit(model)
    trainer.test()
