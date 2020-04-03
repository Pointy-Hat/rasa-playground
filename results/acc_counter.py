import os
import json

def count_acc(dirname):
    js = json.load(open(os.path.join(dirname,"intent_errors.json"), "r"))
    total = len(js)
    correct = 0
    for example in js:
        if example["intent"] == example["intent_prediction"]["name"]:
            correct += 1

    accuracy = correct/total
    with open(os.path.join(dirname,"acc_report"), "w+") as f:
        f.write(str(accuracy))

if __name__=="__main__":
    for d in os.listdir("."):
        if os.path.isdir(d):
            count_acc(d)