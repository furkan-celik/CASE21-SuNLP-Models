import torch
import json
import subprocess
"""
using scorch for CoNLL coreference evaluation. 
"""
def links(pairs,outputs,labels,threshold,set_):
    cc = (labels.flatten()== 1.0).nonzero(as_tuple=False).flatten()
    c = ((outputs.flatten()>threshold).float() == 1.0).nonzero(as_tuple=False).flatten()

    key = list(map(pairs.__getitem__,cc))
    response = list(map(pairs.__getitem__,c))
    
    return key,response,set_    

def return_metrics(key,response,set_):
    
    with open("gold.json","w") as f:
        json.dump({"type":"graph","mentions":list(set([i for j in key for i in j])),"links":key},f)
    with open("sys.json","w") as f:
        json.dump({"type":"graph","mentions":list(set([i for j in response for i in j])),"links":response},f)
        
    subprocess.run(["scorch","gold.json","sys.json","results.txt"])
    
    with open("results.txt","r") as f:
        lines = f.readlines()
        results = [[float(j.split("=")[1]) for j in i.split(":")[1].strip().split("\t") ]for i in lines[:-1]]
        results = results + [float(lines[-1].split(":")[1].strip())]

        muc = dict(zip([set_+"_"+"muc_precision",set_+"_"+"muc_recall",set_+"_"+"muc_f1"],results[0]))
        b_cubed = dict(zip([set_+"_"+"b_cubed_precision",set_+"_"+"b_cubed_recall",set_+"_"+"b_cubed_f1"],results[1]))
        ceaf_e = dict(zip([set_+"_"+"ceaf_e_precision",set_+"_"+"ceaf_e_recall",set_+"_"+"ceaf_e_f1"],results[3]))
        blanc = dict(zip([set_+"_"+"blanc_precision",set_+"_"+"blanc_recall",set_+"_"+"blanc_f1"],results[4]))
        conll = {set_+"_"+"conll":results[-1]}

        metrics = {**muc,**b_cubed,**ceaf_e,**blanc,**conll}
        
    subprocess.run(["rm","gold.json","sys.json","results.txt"])
    
    return metrics

def evaluate(pairs,outputs,labels,threshold,set_):
    return return_metrics(*links(pairs,outputs,labels,threshold,set_))