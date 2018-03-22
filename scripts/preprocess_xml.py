from sys import argv
from numpy import random
import os
import numpy
import subprocess
import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict
dataP=[0.70,0.15,0.15] # how much of the data goes relatively into train, validation and testset in this order.
senN=0 # total number of sentences, used for dividing later
def extract_data(in_path,out_path,lang):
    """
    loops over all files with .tag extension in path recursively and extract from their
    xml tree the sentences and postags into two aligned files.
    1. x_all: will hold sentences line by line.
    2. y_all: will hold morphologie features for each word of each sentence. Features for each sentence in a new line.
    """
    skip="" # which language files to skip/ignore/exclude
    if lang=="nl":
        skip="vl"
    elif lang=="vl":
        skip="nl"
    with open(os.path.join(out_path, "x_all_"+lang), "w") as x_all:
        with open(os.path.join(out_path, "y_all_"+lang), "w") as y_all:
            for subdir, dirs, files in os.walk(in_path, topdown=True):
                dirs[:] = [d for d in dirs if d not in skip]
                for filename in files:
                    if filename.endswith(".tag.gz"):
                        currentFile = os.path.join(subdir, filename)
                        bashCmd= "gzip -fd " + currentFile
                        subprocess.Popen(bashCmd.split(), stdout=subprocess.PIPE)
                for filename in files:
                    if filename.endswith(".tag"):
                        currentFile = os.path.join(subdir, filename)
                        tree = ET.parse(currentFile)
                        root = tree.getroot()
                        global senN # regard as the global senN
                        for i,pau in enumerate(root):
                            senN+=1
                            x_curr=[]
                            y_curr=[]
                            for token in pau:
                                x_curr.append(token.attrib["w"])
                                y_curr.append(token.attrib["pos"])
                                print(" ".join(x_curr), file=x_all)
                                print(" ".join(y_curr), file=y_all)
def divide_data(out_path, lang):
    """
    divides x_all and y_all into train-, validation- and testset with the ratio of amount of sentences
    in each defined by dataP respectively.
    i.e. dataP=[0.70,0.15,0.15] means that the trainset will have 70% of the sentences and validation and test each will have 15% of them.
    """
    file_name=["trn","vld","tst"] # train, validation or test set
    maxSet={fileName:int(p*senN) for fileName,p in zip(file_name, dataP)} # maximum number of sentences in each set
    with open(os.path.join(out_path, "x_all_" + lang), "r") as x_all:
        with open(os.path.join(out_path, "y_all_" + lang), "r") as y_all:
            with open(os.path.join(out_path, "x_trn_" + lang), "w") as x_trn:
                with open(os.path.join(out_path, "y_trn_" + lang), "w") as y_trn:
                    with open(os.path.join(out_path, "x_vld_" + lang), "w") as x_vld:
                        with open(os.path.join(out_path, "y_vld_" + lang), "w") as y_vld:
                            with open(os.path.join(out_path, "x_tst_" + lang), "w") as x_tst:
                                with open(os.path.join(out_path, "y_tst_" + lang), "w") as y_tst:
                                    fileDic={
                                        "trn":(x_trn, y_trn),
                                        "vld":(x_vld, y_vld),
                                        "tst":(x_tst, y_tst)
                                    }
                                    fileCntDic= defaultdict(int)
                                    whileState=True
                                    for x_curr, y_curr in zip(x_all,y_all):
                                        x_curr=x_curr.strip()
                                        y_curr=y_curr.strip()
                                        #decide to which file to write the current sentence by size ratio of train, validation and testset.
                                        while whileState:
                                            key = file_name[random.choice(numpy.arange(3), p=dataP)]
                                            if fileCntDic[key] <= maxSet[key]: break
                                        if fileCntDic["vld"] == maxSet["vld"] and \
                                            fileCntDic["tst"] == maxSet["tst"]:
                                            # in case both test and validation
                                            # sets are full stop guessing the
                                            # key and fill train set with rest.
                                            key = "trn"
                                            whileState=False
                                        fileCntDic[key]+=1
                                        print(x_curr, file=fileDic[key][0])
                                        print(y_curr, file=fileDic[key][1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="Choose language [nl,vl,nl+vl]", default="nl")
    parser.add_argument("--in_path", help="Choose input path (recursive)", default="./")
    parser.add_argument("--out_path", help="Choose output path", default="./")
    args = parser.parse_args()

    extract_data(args.in_path,args.out_path,args.lang)
    divide_data(args.out_path,args.lang)
if __name__ == "__main__":
    main()
