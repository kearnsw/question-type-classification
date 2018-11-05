import re
import sys
import json
import ast
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class PredictionReader:
    def __init__(self, prediction_file):
        self.instances = []
        self.input = None
        self.prediction = None
        self.load(prediction_file)

    def load(self, prediction_file):
        with open(prediction_file) as f:
            for line in f:
                if re.match("input", line):
                    self.input = ast.literal_eval(re.search("{.*?}", line)[0])
                elif re.match("prediction", line) and self.input["AT"] is not "":
                    self.prediction = json.loads(re.search("{.*?}", line)[0])
                    if "encoding" in self.prediction:
                        self.instances.append((self.input["AT"], self.prediction["label"], self.input["Question"], self.prediction["encoding"]))
                    else:
                        self.instances.append((self.input["AT"], self.prediction["label"], self.input["Question"]))
                    self.reset()

    def reset(self):
        self.input = None
        self.prediction = None


if __name__ == "__main__":
    plot = True
    data = PredictionReader(sys.argv[1])
    for instance in data.instances:
        if instance[0] != instance[1]:
        #            print({"true": instance[0], "pred": instance[1], "question": instance[2]})
            print("{0}\t{1}\t{2}".format(instance[0], instance[1], instance[2]))
    y_true, y_pred, question, encoding = zip(*data.instances)
    cm = ConfusionMatrix(y_true, y_pred)
    print(cm)
    if plot:
        cm.plot(normalized=True)
        plt.savefig("normalized_confusion_matrix")

    report = classification_report(y_true, y_pred)
    print(report)
