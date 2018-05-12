"""converts label files into space separated instead of TAG(morphFeature1,morphFeature2)"""
from sys import argv
from collections import defaultdict
from os import rename
def convert():
	if argv[1]==argv[2]:
		# work on temp file if in and out files are equal
		out=argv[2]+".tmp"
	else:
		out=argv[2]
	with open(argv[1], "r") as inF:
		with open(out, "w") as outF:
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
				print(" ".join(newLine).strip(),file=outF)
	if out != argv[2]:
		rename(out,argv[2])
if __name__ == "__main__":
    convert()
