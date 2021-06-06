import re

def func1(sent):
    if " Posted : " in sent:
        pattern = "Posted : \w{3} \w{3} \d{1,2} \d{4} IST ([A-Z]+) ([A-Z]+ )?, (\w+) \d{1,2} :"
        sent = re.sub(pattern,".",sent)
    return sent

def func2(sent):
    if re.search(" IST [A-Za-z]+ :",sent):
        pattern = ".+(January|February|March|May|June|July|August|October|November|December|September|April) \d{2} , \d{4} \d{2}:\d{2} IST [A-Za-z]+ : "
        sent = re.sub(pattern,"",sent)    
    elif " IST " in sent:
        pattern = ".+(January|February|March|May|June|July|August|October|November|December|September|April) \d{2} , \d{4} \d{2}:\d{2} IST "
        sent = re.sub(pattern,"",sent)
    return sent

def func3(sent):
    if "- Indian Express " in sent:
        pattern = "- Indian Express .+ \w{3} \w{3} \d{1,2} \d{4} , \d{2}:\d{2} hrs"
        sent = re.sub(pattern,".",sent)
    return sent

def func4(sent):
    if "PUBLISHED" in sent:
        pattern = "PUBLISHED : \w+ , \d{1,2} (January|February|March|May|June|July|August|October|November|December|September|April) , \d{4} , \d{1,2}:\d{2}(a|p)m"
        sent = re.sub(pattern,".",sent)
    return sent

def func5(sent):
    pattern = ".+\d{1,2}\w\w (January|February|March|May|June|July|August|October|November|December|September|April) \d{4} \d{2}:\d{2} (A|P)M [A-Z]+ : "
    sent = re.sub(pattern,"",sent)
    return sent

def clean_sentence(sent):
    return func5(func4(func3(func2(func1(sent)))))