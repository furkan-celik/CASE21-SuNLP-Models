import os
import json
import random
from itertools import combinations as comb
from transformers import AutoTokenizer
import torch
import pandas as pd
from preprocess import *


def sort_by_index(sent_list):
    return sorted(sent_list,key=lambda x: int(x.split("_")[1]))
def parsedata(path):
    data = {}
    sents= {}
    with open(path,"r") as file:
        idd = 0
        for line in file:
            doc = json.loads(line)
            try: 
                doc_id    = str(doc["id"])
            except:
                doc_id    = str(idd)
                idd +=1
            doc_events= [sort_by_index([doc_id+"_"+str(__) for __ in _]) for _ in doc["event_clusters"]]
            doc_sentno= sort_by_index([doc_id+"_"+str(_) for _ in doc["sentence_no"]])
            for sentno,sent in zip(doc_sentno,doc["sentences"]):
                sents[sentno]= clean_sentence(sent).lower() if sentno in ["1",1] else sent.lower()
            
            data[doc_id]={"events":doc_events,"sents":doc_sentno}
    return data,sents


def createpairs(data,sents,cross=False,ratio=None):
    allpairs_code = []
    positive = []
    pair_idx = {}
    for doc in data:
        allpairs_code+=list(map(list,comb(data[doc]["sents"],2)))
        positive+=sum([list(map(list,list(comb(_,2)))) for _ in data[doc]["events"]],[])
        
    for idx,p in enumerate(allpairs_code):
        pair_idx['_'.join(p)]=idx
        allpairs_code[idx] = [idx]+p+[0.0]
        
    for p in positive:
        index = pair_idx['_'.join(p)]
        allpairs_code[index][-1] = 1.0
        
    if cross:
        allpairs_code_ = [i[1:-1] for i in allpairs_code] 
        total_num = int(len(positive)/ratio)
        missing_num = total_num-len(allpairs_code)+1
        all_ = sorted(list(set([j for i in allpairs_code for j in i[1:-1]])))
        all_p= list(map(list,(random.sample(list(comb(all_,2)),missing_num*2))))
        counter = len(allpairs_code)
        for i in all_p:
            if i not in positive and i not in allpairs_code_:
                allpairs_code.append([counter]+i+[0])
                counter+=1
            
            if counter == total_num:
                break
                
    allpairs_code = [[idx]+s[1:] for idx,s in enumerate(list(random.sample(allpairs_code,len(allpairs_code))))]
    allpairs_sent = [[sents[i] for i in s[1:-1]]+s[-1:] for s in allpairs_code]
    
    
    return allpairs_code,allpairs_sent

def gimme_tensors(sents,tokenizer):
    II, AM, LABEL= [],[],[]
    for i,_ in enumerate(sents):
        label = sents[i][-1]
        text = "</s>".join(sents[i][1:-1])
        encoded = tokenizer.encode_plus(text,                     
                                        add_special_tokens = False,
                                        truncation=True,
                                        max_length=256,
                                        padding="max_length",
                                        return_tensors = 'pt')

        input_id = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        II.append(input_id)
        AM.append(attention_mask)
        LABEL.append(label)
    II = torch.stack(II).squeeze(1)
    AM = torch.stack(AM).squeeze(1)
    LABEL =  torch.tensor(LABEL).view(-1,1).to(torch.float32)
    
    return II,AM,LABEL

def gimme_datasets(DATA_SEED,
                   TRAINPATH,DEVPATH,TESTPATH,
                   TRAIN_CROSS,DEV_CROSS,TEST_CROSS,
                   TRAIN_RATIO,DEV_RATIO,TEST_RATIO,
                   MODEL):
    
    random.seed(DATA_SEED)
    
    train_codes,train_sents=createpairs(*parsedata(TRAINPATH),TRAIN_CROSS ,TRAIN_RATIO)
    dev_codes ,dev_sents =createpairs(*parsedata(DEVPATH ),DEV_CROSS,DEV_RATIO)
    test_codes ,test_sents =createpairs(*parsedata(TESTPATH ),TEST_CROSS,TEST_RATIO)
    
    pd.DataFrame(train_sents).to_csv("../CSVs/TRAIN_SENTS.csv")
    pd.DataFrame(dev_sents).to_csv("../CSVs/DEV_SENTS.csv")
    pd.DataFrame(test_sents).to_csv("../CSVs/TEST_SENTS.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    TR_II,TR_AM,TR_LABEL = gimme_tensors(train_sents,tokenizer)
    DE_II,DE_AM,DE_LABEL = gimme_tensors(dev_sents,tokenizer)
    TE_II,TE_AM,TE_LABEL = gimme_tensors(test_sents,tokenizer)
    
    train_codes,dev_codes,test_codes = [p[1:-1] for p in train_codes],[p[1:-1] for p in dev_codes],[p[1:-1] for p in test_codes]

    return [TR_II,TR_AM,TR_LABEL,DE_II,DE_AM,DE_LABEL,TE_II,TE_AM,TE_LABEL],[train_codes,dev_codes,test_codes]