"""compares the lengths of the lines in two files"""
from sys import argv
def compare():
    with open(argv[1], "r") as frst:
        with open(argv[2], "r") as scnd:
            for i,(line_1,line_2) in enumerate(zip(frst,scnd)):
                len1=len(line_1.strip().split())
                len2=len(line_2.strip().split())
                if len1 != len2:
                    print(i,"\n",line_1,line_2)


if __name__ == "__main__":
    compare()
