import re
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
                elif re.match("prediction", line):
                    self.prediction = json.loads(re.search("{.*?}", line)[0])
                    self.instances.append((self.input["QT"], self.prediction["label"], self.input["Question"]))
                    self.reset()

    def reset(self):
        self.input = None
        self.prediction = None


if __name__ == "__main__":
    plot = False
    data = PredictionReader("../../yahoo_predictions.txt")
    for instance in data.instances:
        if instance[0] != instance[1]:
            print({"true": instance[0], "pred": instance[1], "question": instance[2]})
    y_true, y_pred, question = zip(*data.instances)
    cm = ConfusionMatrix(y_true, y_pred)
    print(cm)
    if plot:
        cm.plot(normalized=True)
        plt.savefig("confusion_matrix")

    report = classification_report(y_true, y_pred)
    print(report)
