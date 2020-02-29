import csv, fasttext, numpy, re, sys

from sklearn.preprocessing import LabelBinarizer
from util import data_io as dio
from nltk import WordNetLemmatizer
from util import scorer

lemmatizer = WordNetLemmatizer()

def normalize(text):
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text=re.sub(r'\W+', ' ', text).strip()
    return text

def tokenize(text):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes text. Returns a list of normalised tokens."""
    text = " ".join(re.split("[^a-zA-Z]*", text.lower())).strip()
    tokens=[]
    for t in text.split():
        if len(t)<4:
            tokens.append(t)
        else:
            tokens.append(lemmatizer.lemmatize(t))
    return tokens

def fit_fasttextt(training_data_json, validation_data_json, class_lvl: int,
                         tmp_folder: str,
                         embedding_file: str):

    if class_lvl==1:
        class_lvl=5
    elif class_lvl==2:
        class_lvl=6
    elif class_lvl==3:
        class_lvl=7
    else:
        print("Not supported")
        exit(1)

    train=numpy.array(dio.read_json(training_data_json))
    val=numpy.array(dio.read_json(validation_data_json))

    X_train=train[:, 1] #use product name only
    y_train = train[:, class_lvl]

    # y_int = encoder.fit_transform(y)
    # y_label_lookup = dict()
    # y_label_lookup_inverse = dict()
    # for index, l in zip(y_int.argmax(1), y):
    #     y_label_lookup[index] = l
    #     y_label_lookup_inverse[l] = index

    X_test = val[:,1]
    y_test = val[:,class_lvl]
    for i in range(len(y_test)):
        label=y_test[i]
        y_test[i]="__label__" + label.replace(" ", "|")

    # prepare fasttext data
    fasttext_train = tmp_folder + "/fasttext_train.tsv"
    with open(fasttext_train, mode='w') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X_train)):
            label = y_train[i]
            text = X_train[i]
            csvwriter.writerow(["__label__" + label.replace(" ", "|"), text])

    if embedding_file is not None:
        model = fasttext.train_supervised(input=fasttext_train,
                                      minn=4, maxn=10, wordNgrams=3,
                                      neg=10, loss='ns', epoch=3000,
                                      thread=30,
                                      dim=300,
                                      pretrainedVectors=embedding_file)
    else:
        model = fasttext.train_supervised(input=fasttext_train,
                                          minn=4, maxn=10, wordNgrams=3,
                                          neg=10, loss='ns', epoch=3000,
                                          thread=30,
                                          dim=300)
    # evaluate the model
    predictions = model.predict(list(X_test))[0]

    return scorer.score(predictions, list(y_test))

if __name__ == "__main__":
    training_data_json=sys.argv[1]
    validation_data_json = sys.argv[2]
    temporary_folder = sys.argv[3]

    sum_p = 0.0
    sum_r = 0.0
    sum_f1 = 0.0
    #lvl1
    p, r, f1, support=fit_fasttextt(training_data_json, validation_data_json,1, temporary_folder, None)
    print("Lvl1 P={} R={} F1={}".format(p, r, f1))
    sum_p += p
    sum_r += r
    sum_f1 += f1

    # lvl2
    p, r, f1, support = fit_fasttextt(training_data_json, validation_data_json, 2, temporary_folder, None)
    print("Lvl2 P={} R={} F1={}".format(p, r, f1))
    sum_p += p
    sum_r += r
    sum_f1 += f1

    # lvl3
    p, r, f1, support = fit_fasttextt(training_data_json, validation_data_json, 3, temporary_folder, None)
    print("Lvl3 P={} R={} F1={}".format(p, r, f1))
    sum_p += p
    sum_r += r
    sum_f1 += f1

    print("Average P={}, R={}, F1={}".format(sum_p / 3, sum_r / 3, sum_f1 / 3))