import copy

def read_files(fname):
    sentences = []

    with open(fname, "r+") as inFile:
        sentence = []
        for line in inFile.readlines()[1:]:
            if "[SEP]" in line:
                sentences.append(copy.deepcopy(sentence) + [line])
                sentence = []
            elif len(line) > 0 and "\t" in line:
                sentence.append(line[:-1].split("\t"))

        if sentence != []:
            sentences.append(copy.deepcopy(sentence) + [line])

    return sentences