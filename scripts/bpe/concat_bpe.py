"""concatenates characters into words, desegementation of the python script
apply_bpe_2.py
args: [input file,output file] """
from sys import argv
def concat(inPath,outPath):
    with open(inPath, "r") as inF:
        with open(outPath, "w") as outF:
            for line in inF:
                line=line.strip().split("~")
                line=" ".join(word.replace(" ","") for word in line)
                print(line, file=outF)

if __name__ == "__main__":
    concat(argv[1],argv[2])
