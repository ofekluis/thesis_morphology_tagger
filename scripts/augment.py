"""adds custom symbol after every token in every line of the file"""
"""args: filename, token, output_filename"""
from sys import argv
def augment():
    sym=argv[2]
    with open(argv[1], "r") as inF:
        with open(argv[3], "w") as outF:
            for line in inF:
                newline=" ".join(token+sym for token in line.strip().split())
                print(newline,file=outF)

if __name__ == "__main__":
    augment()
