"""converts file from one sentence in a line to one token/word in line"""
from sys import argv
def convert():
    with open(argv[1], "r") as inF:
        with open(argv[2], "w") as outF:
            for line in inF:
                for token in line.strip().split(" "):
                    print(token, file=outF)


if __name__ == "__main__":
    convert()
