"""gets accuracy of marmot predicted file compared to gold standard file from corpus (both should be with each token/word in a new line"""
from sys import argv
def get_accuracy():
    same=0
    count=0
    with open(argv[1], "r") as real:
        with open(argv[2], "r") as pred:
            for i,(line_r,line_p) in enumerate(zip(real,pred)):
                line_r=line_r.strip().split("(")
                line_p=line_p.strip().split()
                pos_r=line_r[0]
                morph_r=line_r[1][:-1].replace(",","|")
                if morph_r=="":
                    morph_r="Non"
                if len(line_p)>0:
                    pos_p=line_p[5]
                    morph_p=line_p[7]
                    if pos_r==pos_p and morph_r==morph_p:
                        count+=1
                        same+=1
                if i%10==0:
                    if count>7:
                        print(i)
                else:
                    count=0
    print(same/(i+1))
if __name__ == "__main__":
    get_accuracy()
