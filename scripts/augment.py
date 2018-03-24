"""adds augmented symbols to each token in every line of both files
the results is a joined file with augmented tokens, lines alternate between languages
can also be used to just concatenate two files (with delibarte mixing of the lines)"""
import argparse
def augment(doc1, doc2, out, sym1, sym2, aug):
    sym= (sym1,sym2) if aug else ("","")
    doc=[doc1,doc2]
    with open(doc[0], "r") as doc[0]:
        with open(doc[1], "r") as doc[1]:
            with open(out, "w") as outF:
                lineLst=[[line.strip().split() for line in d.readlines()] for d in doc]
                for i,lines in enumerate(zip(lineLst[0],lineLst[1])):
                    for line,s in zip(lines,sym):
                        line=" ".join(s+token+s for token in line)
                        print(line, file=outF)
                # add left lines from the longer file
                rest=1 if i>len(lineLst[0]) else 0
                for j in range(i+1,len(lineLst[rest])):
                    line=" ".join(sym[rest]+token+sym[rest] for token in lineLst[rest][j])
                    print(line,file=outF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc1", help="Choose first document to join", default="x_trn_nl")
    parser.add_argument("--doc2", help="Choose second document to join", default="x_trn_vl")
    parser.add_argument("--sym1", help="Choose augmentation symbol for first document's tokens", default="*")
    parser.add_argument("--sym2", help="Choose augmentation symbol for second document's tokens", default="^")
    parser.add_argument("--out", help="Choose joined document path/name", default="./aug")
    parser.add_argument("--aug", help="Should documents be augmented or only joined?", default=True)
    args = parser.parse_args()
    if args.aug=="false" or args.aug=="False":
        args.aug=False
    augment(args.doc1,args.doc2,args.out,args.sym1,args.sym2,args.aug)
