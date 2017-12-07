import sys
import os
import re
import math
import collections
import random
from copy import deepcopy
import matplotlib.pyplot as plt

header = []

class Perceptron: 
    def __init__(self, feature_num):
        self.output = None
        self.weights = [None] * (feature_num + 1)
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)
            while self.weights[i] == -1:
                self.weights[i] = random.uniform(-1, 1)

class NeuralNetwork:
    def __init__(self, feature_num, learning_rate):
        self.learning_rate = learning_rate
        self.output_layer = Perceptron(feature_num)
        self.hidden_layer = [];
        for i in range(feature_num):
            self.hidden_layer.append(Perceptron(feature_num))

    def calOutputs(self, train_sample):
        for h_idx in range(len(self.hidden_layer)):
            output = 0.0
            for w_idx in range(len(self.hidden_layer[0].weights)):
                if w_idx == 0:
                    output = self.hidden_layer[h_idx].weights[0]
                else:
                    output += self.hidden_layer[h_idx].weights[w_idx] * train_sample[w_idx - 1]
            self.hidden_layer[h_idx].output = self.sigmoid(output)

        for w_idx in range(len(self.output_layer.weights)):
            if w_idx == 0:
                output = self.output_layer.weights[0]
            else:
                output += self.output_layer.weights[w_idx] * self.hidden_layer[w_idx - 1].output
        self.output_layer.output = self.sigmoid(output)

    def updateWeight(self, train_sample):
        if header[-1][1] == train_sample[-1]:
            common_item = -self.learning_rate * (self.output_layer.output - 0)
        else:
            common_item = -self.learning_rate * (self.output_layer.output - 1)

        # Update weights of hidden layer
        for h_idx in range(len(self.hidden_layer)):
            for w_idx in range(len(self.hidden_layer[h_idx].weights)):
                if w_idx == 0:
                    wgt_gd = common_item * self.output_layer.weights[h_idx + 1] * self.hidden_layer[h_idx].output * (1 - self.hidden_layer[h_idx].output)
                else:
                    wgt_gd = common_item * self.output_layer.weights[h_idx + 1] * self.hidden_layer[h_idx].output * (1 - self.hidden_layer[h_idx].output) * train_sample[w_idx - 1]
            self.hidden_layer[h_idx].weights[w_idx] += wgt_gd

        # Update weights of output layer
        for i in range(len(self.output_layer.weights)):
            if i == 0:
                wgt_gd = common_item
            else:
                wgt_gd = common_item * self.hidden_layer[i - 1].output
            self.output_layer.weights[i] += wgt_gd

    def sigmoid(self, val):
        try:
            res = (1.0 / (1 + math.pow(math.e, -val)))
            return res
        except OverflowError:
            print("val = {}".format(val))
            res = (1.0 / (1 + math.pow(math.e, -val)))
            print("res = {}".format(res))

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def parseARFF(file_name):
    infile = open(file_name, "r")
    header_data_idx = 0
    file_data_idx = 0
    header_data = []
    file_data = []

    while True:
        line = infile.readline()
        if not line:
            break
        match1 = re.search("@relation", line)
        match2 = re.search("@attribute", line)
        match3 = re.search("@data", line)
        match4 = re.search("%", line)
        
        line = line.rstrip("\n")

        if not (match2 is None):
            header_data.append([])
            line = re.split('@attribute|\'|,| |{|}', line)
            line = filter(None, line)
            header_data[header_data_idx] = line
            header_data_idx += 1
        elif (match1 is None) and (match3 is None) and (match4 is None):
            file_data.append([])
            new_line = re.split(',', line)

            for i in range(len(new_line)):
                if isFloat(new_line[i]):
                    new_line[i] = float(new_line[i])
                
                file_data[file_data_idx].append(new_line[i])

            file_data_idx += 1

    return header_data, file_data

def separateTrainSample(header, train_sample):
    pos_train_sample = []
    neg_train_sample = []

    for t in train_sample:
        if header[-1][1] == t[-1]:
            neg_train_sample.append(t)
        else:
            pos_train_sample.append(t)

    return pos_train_sample, neg_train_sample

def SGD(feature_num, train_sample, num_epochs, learning_rate):
    # Initialize all weights of all perceptrons in neural network to [-1, 1)
    nn = NeuralNetwork(feature_num, learning_rate)

    for i in range(num_epochs):
        for t in train_sample:
            # Input training sample to the network and compute all perceptrons' outputs
            nn.calOutputs(t)

            # Update weights based on gradient descent theory
            nn.updateWeight(t)

    return nn

def predictTestSample(all_test_results, test_sample, fold_of_instance, nn):
    for i in range(len(test_sample)):
        nn.calOutputs(test_sample[i])
        idx_of_total = test_sample[i][-2]

        all_test_results[idx_of_total] = []
        all_test_results[idx_of_total].append(fold_of_instance)

        if nn.output_layer.output < 0.5:
            all_test_results[idx_of_total].append(header[-1][1])
        else:
            all_test_results[idx_of_total].append(header[-1][2])

        all_test_results[idx_of_total].append(test_sample[i][-1])
        all_test_results[idx_of_total].append(nn.output_layer.output)

def doSGD(all_pos_train_sample, all_neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate):
    each_fold_size = int(float(total_train_size) / num_folds + 0.5)
    pos_ratio = float(len(all_pos_train_sample)) / total_train_size
    each_fold_pos_size = int(float(each_fold_size) * pos_ratio + 0.5)
    each_fold_neg_size = each_fold_size - each_fold_pos_size

    all_test_results = [None] * (total_train_size)

    random.shuffle(all_pos_train_sample)
    random.shuffle(all_neg_train_sample)

    for n in range(num_folds):
        pos_test_start_idx = n * each_fold_pos_size
        neg_test_start_idx = n * each_fold_neg_size

        if n == num_folds - 1:
            pos_test_end_idx = len(all_pos_train_sample) - 1
            neg_test_end_idx = len(all_neg_train_sample) - 1
        else:
            pos_test_end_idx = (n + 1) * each_fold_pos_size - 1
            neg_test_end_idx = (n + 1) * each_fold_neg_size - 1

        pos_test_sample = all_pos_train_sample[pos_test_start_idx : (pos_test_end_idx + 1)]
        neg_test_sample = all_neg_train_sample[neg_test_start_idx : (neg_test_end_idx + 1)]

        test_sample = pos_test_sample + neg_test_sample;
        random.shuffle(test_sample)

        pos_train_sample = all_pos_train_sample[0 : pos_test_start_idx] + all_pos_train_sample[(pos_test_end_idx + 1) : ]
        neg_train_sample = all_neg_train_sample[0 : neg_test_start_idx] + all_neg_train_sample[(neg_test_end_idx + 1) : ]

        train_sample = pos_train_sample + neg_train_sample;
        random.shuffle(train_sample)

        feature_num = len(train_sample[0]) - 2

        nn = SGD(feature_num, train_sample, num_epochs, learning_rate)

        predictTestSample(all_test_results, test_sample, n, nn)

    return all_test_results

def plotPartB1(pos_train_sample, neg_train_sample, total_train_size):
    epoch_list = [25, 50, 75, 100]
    avg_accuracy_list = []

    for e in epoch_list:
        match_num = 0;
        all_test_results = doSGD(pos_train_sample, neg_train_sample, total_train_size, 10, e, 0.1)

        for a in all_test_results:
            if (a[1] == a[2]):
                match_num += 1

        avg_accuracy_list.append(float(match_num) / total_train_size)

    new_x, new_y = zip(*sorted(zip(epoch_list, [i * 100 for i in avg_accuracy_list])))
    plt.plot(new_x, new_y, 'go--', label='avg accuracy')

    global_min = max(min([i * 100 for i in avg_accuracy_list]) - 1, 0)
    global_max = min(max([i * 100 for i in avg_accuracy_list]) + 1, 100)
    plt.axis([0, epoch_list[-1], global_min, global_max])
    plt.xticks(new_x)
    plt.yticks(new_y)
    plt.grid(which='both')
    plt.xlabel('# of epoch')
    plt.ylabel('% of accuracy')
    plt.title("Part B 1")
    l = plt.legend(loc = 4)
    plt.show()

def plotPartB2(pos_train_sample, neg_train_sample, total_train_size):
    num_folds_list = [5, 10, 15, 20, 25]
    avg_accuracy_list = []

    for n in num_folds_list:
        match_num = 0;
        all_test_results = doSGD(pos_train_sample, neg_train_sample, total_train_size, n, 50, 0.1)

        for a in all_test_results:
            if (a[1] == a[2]):
                match_num += 1

        avg_accuracy_list.append(float(match_num) / total_train_size)

    new_x, new_y = zip(*sorted(zip(num_folds_list, [i * 100 for i in avg_accuracy_list])))
    plt.plot(new_x, new_y, 'go--', label='avg accuracy')

    global_min = max(min([i * 100 for i in avg_accuracy_list]) - 1, 0)
    global_max = min(max([i * 100 for i in avg_accuracy_list]) + 1, 100)
    plt.axis([0, num_folds_list[-1], global_min, global_max])
    plt.xticks(new_x)
    plt.yticks(new_y)
    plt.grid(which='both')
    plt.xlabel('# of folds')
    plt.ylabel('% of accuracy')
    plt.title("Part B 2")
    l = plt.legend(loc = 4)
    plt.show()

def plotPartB3(pos_train_sample, neg_train_sample, total_train_size):
    all_test_results = doSGD(pos_train_sample, neg_train_sample, total_train_size, 10, 50, 0.1)
    all_test_results.sort(key = lambda x : (1 / x[-1]))

    num_neg = len(neg_train_sample)
    num_pos = len(pos_train_sample)

    TP = 0
    FP = 0
    last_TP = 0
    neg = header[-1][1]
    pos = header[-1][2]
    FPR_list = []
    TPR_list = []

    for i in range(len(all_test_results)):
        if (i > 0) and (all_test_results[i][-1] != all_test_results[i - 1][-1]) and (all_test_results[i][2] == neg) and (TP > last_TP):
            FPR = float(FP) / num_neg
            TPR = float(TP) / num_pos
            FPR_list.append(FPR)
            TPR_list.append(TPR)

            last_TP = TP

        if all_test_results[i][2] == pos:
            TP += 1
        else:
            FP += 1

    FPR = float(FP) / num_neg
    TPR = float(TP) / num_pos
    FPR_list.append(FPR)
    TPR_list.append(TPR)

    new_x, new_y = zip(*sorted(zip(FPR_list, TPR_list)))
    plt.plot(new_x, new_y, 'go--')
    neutral_lin = [float(i) / 100 for i in range(101)]
    plt.plot(neutral_lin, neutral_lin, 'k--')

    plt.axis([0, 1.0, 0, 1.0])
    plt.grid(which='both')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("Part B 3")
    plt.show()

def main():
    assert len(sys.argv) == 5

    # Part A - Programming
    global header
    header, train_sample = parseARFF(sys.argv[1])

    num_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

    # Label training samples for outputing prediction results in the original order
    for i in range(len(train_sample)):
        train_sample[i].append(train_sample[i][-1])
        train_sample[i][-2] = i

    total_train_size = len(train_sample)
    pos_train_sample, neg_train_sample = separateTrainSample(header, train_sample)

    all_test_results = doSGD(pos_train_sample, neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate)

    # Print outputs in the following order: fold_of_instance predicted_class actual_class confidence_of_prediction
    for a in all_test_results:
        print("{} {} {} {}".format(a[0], a[1], a[2], a[3]))

    # Part B 1 - Programming
    #plotPartB1(pos_train_sample, neg_train_sample, total_train_size)

    # Part B 2 - Programming
    #plotPartB2(pos_train_sample, neg_train_sample, total_train_size)

    # Part B 3 - Programming
    #plotPartB3(pos_train_sample, neg_train_sample, total_train_size)

if __name__ == "__main__":
    main()