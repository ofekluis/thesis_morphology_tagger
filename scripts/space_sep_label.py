"""converts label files into space separated instead of TAG(morphFeature1,morphFeature2)"""
from sys import argv
from collections import defaultdict
def convert():
	with open(argv[1], "r") as inF:
		with open(argv[2], "w") as outF:
			for line in inF:
				line=line.strip().split()
				newLine=[]
				for token in line:
					token=token.split("(")
					tag=token[0]
					morph= token[1][:-1].replace(","," ")
					if morph:
						newLine.append(tag + " " + morph)
					else:
						newLine.append(tag)
				print(" ".join(newLine),file=outF)
if __name__ == "__main__":
    convert()
