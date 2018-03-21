"""converts token and label files into marmot capable format"""
from sys import argv
from collections import defaultdict
def convert():
    #dic=defaultdict(lambda: defaultdict(lambda: set())) # dic[word][pos_tag]={set of morph_tags}
    with open(argv[1], "r") as x:
        with open(argv[2], "r") as y:
            with open("./marmot_train", "w") as out:
                for x_seq,y_seq in zip(x,y):
                    x_seq=x_seq.strip().split()
                    y_seq=y_seq.strip().split()
                    for token,label in zip(x_seq,y_seq):
                        label=label.split("(")
                        pos=label[0]
                        morph=label[1][:-1].replace(",","|")
                        if morph=="":
                            morph="Non"
                        print(token,pos,morph,file=out)


if __name__ == "__main__":
    convert()
