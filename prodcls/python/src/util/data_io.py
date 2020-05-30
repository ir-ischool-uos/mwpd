import json
import operator
import pandas as pd
import sys

''''
This method reads the json data file (train/val/test) and save them as a matrix where each row is an instance with the following columns:
- 0: id
- 1: name
- 2: description
- 3: categorytext
- 4: url
- 5: lvl1
- 6: lvl2
- 7: lvl3
'''
def read_json(in_file):
    matrix=[]
    with open(in_file) as file:
        line = file.readline()

        while line is not None and len(line)>0:
            js=json.loads(line)

            row=[js['ID'],js['Name'],js['Description'],js['CategoryText'],js['URL'],js['lvl1'],js['lvl2'],js['lvl3']]
            matrix.append(row)
            line=file.readline()
    return matrix


'''
Utility method to remove labels from GS datasets (must conform to the json format)
'''
def remove_labels_json(in_file, out_file):
    f = open(out_file, "w")
    with open(in_file) as file:
        line = file.readline()

        while line is not None and len(line)>0:
            js=json.loads(line)
            del js['lvl1']
            del js['lvl2']
            del js['lvl3']
            f.write(json.dumps(js)+"\n")
            line = file.readline()
    f.close()


''''
This method reads the test data output file CSV and save them as a matrix where each row is an instance with the following columns:
- 0: id
- 1: lvl1
- 2: lvl2
- 3: lvl3
'''
def read_csv(in_file):
    df = pd.read_csv(in_file, header=None, delimiter=',', quoting=0, encoding="utf-8")
    df=df.fillna('None')
    return df.to_numpy()

'''
output a matrix in the above format to json
'''
def write_json(matrix, out_file):
    freq=dict()
    with open(out_file,'w') as file:
        for row in matrix:
            data=dict()
            data["ID"]=row[0]
            data["Name"] = row[1]
            data["Description"] = row[2]
            data["CategoryText"] = row[3]
            data["URL"] = row[4]
            data["lvl1"] = row[5]
            data["lvl2"] = row[6]
            data["lvl3"] = row[7]
            js=json.dumps(data)

            file.write(js+"\n")

            if row[7] in freq.keys():
                freq[row[7]]+=1
            else:
                freq[row[7]]=1

    sorted_x = sorted(freq.items(), key=operator.itemgetter(1))
    for t in sorted_x:
        print("lvl3 class {}={}".format(t[0],t[1]))
    print("\n")


if __name__ == "__main__":
    remove_labels_json(sys.argv[1],
                       sys.argv[2])