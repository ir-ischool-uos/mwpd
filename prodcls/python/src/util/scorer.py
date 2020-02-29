'''
This file defines methods for scoring the output

Check the '__main__' section of this class and edit the code as you need to,
then run the file to score your output
'''

import sys, numpy
from sklearn.metrics import precision_recall_fscore_support
from util import data_io as dio

def read_prediction(in_file, format):
    if format=='json':
        data= numpy.array(dio.read_json(in_file))
        return data[:, 5], data[:, 6], data[:, 7]
    elif format=='csv':
        data= dio.read_json(in_file)
        return data[:, 1], data[:, 2], data[:, 3]
    else:
        print("Not supported input format")
        return None

def read_gold_standard(in_file):
    data= numpy.array(dio.read_json(in_file))
    return data[:,5],data[:,6],data[:,7]

def score(prediction:list, gs:list):
    return precision_recall_fscore_support(gs,prediction, average='weighted')


if __name__ == "__main__":
    #gold standard must conform to the json format of the train/val sets
    gold_standard_file=sys.argv[1]
    #prediction can take two different formats, see below
    prediction_file=sys.argv[2]

    #if 'json' the format should comform to the json format of the train/val sets
    #if 'csv' the format should conform to the required CSV format for system output
    prediction_format=sys.argv[3]

    pred_lvl1, pred_lvl2, pred_lvl3=read_prediction(prediction_file, prediction_format)
    gs_lvl1, gs_lvl2, gs_lvl3=read_gold_standard(gold_standard_file)

    sum_p=0.0
    sum_r=0.0
    sum_f1=0.0

    p,r, f1, support=score(list(pred_lvl1), list(gs_lvl1))
    print("Lvl1 P={} R={} F1={}".format(p,r, f1))
    sum_p+=p
    sum_r+=r
    sum_f1+=f1
    p, r, f1, support = score(pred_lvl2, gs_lvl2)
    print("Lvl2 P={} R={} F1={}".format(p,r,f1))
    sum_p += p
    sum_r += r
    sum_f1 += f1
    p, r, f1, support = score(pred_lvl3, gs_lvl3)
    print("Lvl3 P={} R={} F1={}".format(p, r, f1))
    sum_p += p
    sum_r += r
    sum_f1 += f1
    print("Average P={}, R={}, F1={}".format(sum_p/3, sum_r/3, sum_f1/3))
