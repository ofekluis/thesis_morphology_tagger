"""adds augmented symbols after every token in every line of both files
the results is a joined file with augmented tokens, lines alternate between languages"
args: filename, token, output_filename"""
from sys import argv
def augment():
    with open(argv[1], "r") as lang1:
        with open(argv[2], "r") as lang2:
            with open(argv[3], "w") as outF:
                lineLst=[[],[]]
                lineLst[0]=[line.strip().split() for line in lang1.readlines()]
                lineLst[1]=[line.strip().split() for line in lang2.readlines()]
                for i,(line1,line2) in enumerate(zip(lineLst[0],lineLst[1])):
                    line1=" ".join("*"+token+"*" for token in line1)
                    line2=" ".join("^"+token+"^" for token in line2)
                    print(line1,file=outF)
                    print(line2,file=outF)
                # add left lines from the longer file
                rest=1 if i>len(lineLst[0]) else 0
                sym=("*","^")[rest]
                for j in range(i+1,len(lineLst[rest])):
                    line2=" ".join(sym+token+sym for token in lineLst[rest][j])
                    print(line2,file=outF)


if __name__ == "__main__":
    augment()
