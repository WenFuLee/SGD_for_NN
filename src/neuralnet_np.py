import sys
import os
import re
import math
import collections
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import numpy as np

header = []

class NeuralNetwork:
    def __init__(self, input_dim, learning_rate, gd_tech):
        hidden_nodes = input_dim / 14
        self.learning_rate = learning_rate
        self.gd_tech = gd_tech
        self.W_matrix = 2*np.random.rand(input_dim + 1, hidden_nodes) - 1
        self.V_matrix = 2*np.random.rand(hidden_nodes + 1) - 1
        self.hidden_out = None
        self.final_out = None

        # for adagrad and adam
        self.sum_dW_square = np.zeros((input_dim + 1, hidden_nodes))
        self.sum_dV_square = np.zeros((hidden_nodes + 1))
        self.sum_dW = np.zeros((input_dim + 1, hidden_nodes))
        self.sum_dV = np.zeros((hidden_nodes + 1))
        self.update_time = 0

    def calOutputs(self, train_sample):
        self.hidden_out = np.dot(np.array([1] + train_sample[:-2]), self.W_matrix)
        self.hidden_out = 1 / (1 + np.exp(-self.hidden_out))
        self.final_out = np.dot(np.hstack((1, self.hidden_out)), self.V_matrix)
        self.final_out = 1 / (1 + np.exp(-self.final_out))

    def updateWeight(self, train_sample):
        if self.gd_tech == 'SGD':
            self.updateWeightSGD(train_sample)
        elif self.gd_tech == 'AdaGrad':
            self.updateWeightAdaGrad(train_sample)
        elif self.gd_tech == 'Adam':
            self.updateWeightAdam(train_sample)
        else:
            print('Wrong argument names for GD techniques!')

    def updateWeightSGD(self, train_sample):
        if header[-1][1] == train_sample[-1]:
            dcdz = self.final_out - 0
        else:
            dcdz = self.final_out - 1

        
        '''     
        print 'dody*hidden', np.shape(dcdz*np.hstack((1, self.hidden_out)))
        print 'all', np.shape(self.learning_rate*dcdz*np.hstack((1, self.hidden_out)))
        print 'all', np.shape(self.learning_rate*np.hstack((1, self.hidden_out))*dcdz)
        '''     
        # Update weights of output layer (V_matrix)
        new_V = self.V_matrix - self.learning_rate * np.hstack((1, self.hidden_out)) * dcdz # 1+hidden, 1

        # Update weights of W_matrix
        gamma = np.multiply( self.V_matrix[1:], np.multiply(self.hidden_out, 1 - self.hidden_out)) * dcdz # hidden, 1
        new_W = self.W_matrix - self.learning_rate * np.outer(np.hstack((1, train_sample[:-2])), gamma) # 1+p, hidden
        '''     
        print "W", np.shape(self.W_matrix)
        print "V", np.shape(self.V_matrix)
        print "newW", np.shape(new_W)
        print "newV", np.shape(new_V)
        '''
        self.V_matrix = new_V
        self.W_matrix = new_W

    def updateWeightAdaGrad(self, train_sample):
        epsilon = math.pow(10, -8)

        if header[-1][1] == train_sample[-1]:
            dcdz = self.final_out - 0
        else:
            dcdz = self.final_out - 1

        # Update weights of output layer
        dcdv = np.hstack((1, self.hidden_out)) * dcdz # 1 + hidden, 1
        self.sum_dV_square += np.power(dcdv, 2)
        new_V = self.V_matrix - self.learning_rate * np.divide(dcdv, np.sqrt(self.sum_dV_square + epsilon))

        # Update weights of hidden layer
        gamma = np.multiply( self.V_matrix[1:], np.multiply(self.hidden_out, 1 - self.hidden_out)) * dcdz # hidden, 1
        dcdw = np.outer(np.hstack((1, train_sample[:-2])), gamma) # 1 + p, hidden
        self.sum_dW_square += np.power(dcdw, 2)
        #new_W = self.W_matrix - self.learning_rate * np.divide(dcdw, np.sqrt(self.sum_dW_square + epsilon))
        new_W = self.W_matrix - self.learning_rate * np.divide(dcdw, np.sqrt(self.sum_dW_square) + epsilon)

        self.V_matrix = new_V
        self.W_matrix = new_W

    def updateWeightAdam(self, train_sample):
        epsilon = math.pow(10, -8)
        beta_1 = 0.9
        beta_2 = 0.999
        self.update_time += 1

        if header[-1][1] == train_sample[-1]:
            dcdz = self.final_out - 0
        else:
            dcdz = self.final_out - 1

        # Update weights of output layer
        dcdv = np.hstack((1, self.hidden_out)) * dcdz # 1 + hidden, 1
        self.sum_dV = beta_1*self.sum_dV + (1 - beta_1)*dcdv
        self.sum_dV_square = beta_2*self.sum_dV_square + (1 - beta_2)*np.power(dcdv, 2)
        # bias correction
        sum_dV_correct = self.sum_dV / (1 - beta_1**self.update_time)
        sum_dV_square_correct = self.sum_dV_square / (1 - beta_2**self.update_time)
        new_V = self.V_matrix - self.learning_rate * np.divide(sum_dV_correct, np.sqrt(sum_dV_square_correct) + epsilon)

        # Update weights of hidden layer
        gamma = np.multiply( self.V_matrix[1:], np.multiply(self.hidden_out, 1 - self.hidden_out)) * dcdz # hidden, 1
        dcdw = np.outer(np.hstack((1, train_sample[:-2])), gamma) # 1 + p, hidden
        self.sum_dW = beta_1*self.sum_dW + (1 - beta_1) * dcdw
        self.sum_dW_square = beta_2*self.sum_dW_square + (1 - beta_2)*np.power(dcdw, 2)
        # bias correction
        sum_dW_correct = self.sum_dW / (1 - beta_1**self.update_time)
        sum_dW_square_correct = self.sum_dW_square / (1 - beta_2**self.update_time)
        new_W = self.W_matrix - self.learning_rate * np.divide(sum_dW_correct, np.sqrt(sum_dW_square_correct) + epsilon)

        self.V_matrix = new_V
        self.W_matrix = new_W

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

def parseMNIST(data_dir):
    from mnist import MNIST
    mndata = MNIST(data_dir)
    images, labels = mndata.load_training()
    for i in range(len(labels)):
        images[i].append(labels[i])
    header_data = [['attribute_%d'%(i+1), 'numeric'] for i in range(len(images[0]))]
    header_class = ['Class']
    header_class.extend(range(10))
    header_data.append(header_class)
    return header_data, images

def parseMNIST_target(data_dir, target, data_size):
    from mnist import MNIST
    from collections import defaultdict

    # file_data
    sz = data_size/18
    mndata = MNIST(data_dir)
    images, labels = mndata.load_training()
    label_to_images = defaultdict(list)
    label_cnt = defaultdict(lambda: 0)
    data_cnt = 0
    for image, label in zip(images, labels):
        if data_cnt >= data_size: break
        if label==target:
            if label_cnt[label] >= sz*9 and data_cnt < sz*18:
                continue
            image = map(lambda x:float(x)/255, image)
            image.append(1)
        else:
            if label_cnt[label] >= sz and data_cnt < sz*18:
                continue
            image = map(lambda x:float(x)/255, image)
            image.append(0)
        label_to_images[label].append(image)
        label_cnt[label] += 1
        data_cnt += 1
    file_data = list()
    for v in label_to_images.values():
        file_data.extend(v)

    # header_data
    header_data = [['attribute_%d'%(i+1), 'numeric'] for i in range(len(images[0]))]
    header_data.append(['Class', 0, 1])
    return header_data, file_data

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

def doGD(feature_num, train_sample, num_epochs, learning_rate, gd_tech):
    # Initialize all weights of all perceptrons in neural network to [-1, 1)
    nn = NeuralNetwork(feature_num, learning_rate, gd_tech)

    for i in range(num_epochs):
        random.shuffle(train_sample)
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

        if nn.final_out < 0.5:
            all_test_results[idx_of_total].append(header[-1][1])
        else:
            all_test_results[idx_of_total].append(header[-1][2])

        all_test_results[idx_of_total].append(test_sample[i][-1])
        all_test_results[idx_of_total].append(nn.final_out)

def getLoss(sample, nn):
    result = 0
    for i in range(len(sample)):
        nn.calOutputs(sample[i])
        if nn.final_out < 1e-8:
            nn.final_out = 1e-8
        if nn.final_out > (1 - 1e-8):
            nn.final_out = (1 - 1e-8)
        cross_loss = - (sample[i][-1]*math.log(nn.final_out) + (1 - sample[i][-1]) * math.log(1 - nn.final_out))
        result += cross_loss
    return cross_loss

def SCV_v3(all_pos_train_sample, all_neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate, gd_tech):
    avg_loss_list = []

    each_fold_size = int(float(total_train_size) / num_folds + 0.5)
    pos_ratio = float(len(all_pos_train_sample)) / total_train_size
    each_fold_pos_size = int(float(each_fold_size) * pos_ratio + 0.5)
    each_fold_neg_size = each_fold_size - each_fold_pos_size

    # Initialize all weights of all perceptrons in neural network to [-1, 1)
    nn_list = []
    for n in range(num_folds):
        random.seed(num_epochs)
        np.random.seed(num_epochs)
        nn = NeuralNetwork(len(header) - 1, learning_rate, gd_tech)
        nn_list.append(nn)

    for i in range(num_epochs):
        time_start = time.clock()

        #all_test_results = [None] * (total_train_size)
        cross_test_loss = 0
        cross_train_loss = 0
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
            #random.shuffle(test_sample)

            pos_train_sample = all_pos_train_sample[0 : pos_test_start_idx] + all_pos_train_sample[(pos_test_end_idx + 1) : ]
            neg_train_sample = all_neg_train_sample[0 : neg_test_start_idx] + all_neg_train_sample[(neg_test_end_idx + 1) : ]

            train_sample = pos_train_sample + neg_train_sample;
            #random.shuffle(train_sample)

            random.shuffle(train_sample)
            for t in train_sample:
                # Input training sample to the network and compute all perceptrons' outputs
                nn_list[n].calOutputs(t)

                # Update weights based on gradient descent theory
                nn_list[n].updateWeight(t)

            cross_test_loss += getLoss(test_sample, nn_list[n])
            cross_train_loss += getLoss(train_sample, nn_list[n])

        time_elapsed = (time.clock() - time_start)
        print('i = {}, time_elapsed = {}'.format(i, time_elapsed))

        print('total_train_size = {}, cross-train-loss = {}, cross-test-loss = {}'.format( \
                total_train_size, cross_train_loss / num_folds, cross_test_loss / num_folds))
        avg_loss_list.append(cross_test_loss / num_folds)

    return avg_loss_list

def SCV(all_pos_train_sample, all_neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate, gd_tech):
    each_fold_size = int(float(total_train_size) / num_folds + 0.5)
    pos_ratio = float(len(all_pos_train_sample)) / total_train_size
    each_fold_pos_size = int(float(each_fold_size) * pos_ratio + 0.5)
    each_fold_neg_size = each_fold_size - each_fold_pos_size

    all_test_results = [None] * (total_train_size)

    #random.shuffle(all_pos_train_sample)
    #random.shuffle(all_neg_train_sample)

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
        #random.shuffle(test_sample)

        pos_train_sample = all_pos_train_sample[0 : pos_test_start_idx] + all_pos_train_sample[(pos_test_end_idx + 1) : ]
        neg_train_sample = all_neg_train_sample[0 : neg_test_start_idx] + all_neg_train_sample[(neg_test_end_idx + 1) : ]

        train_sample = pos_train_sample + neg_train_sample;
        #random.shuffle(train_sample)

        feature_num = len(train_sample[0]) - 2

        random.seed(num_epochs)
        np.random.seed(num_epochs)
        nn = doGD(feature_num, train_sample, num_epochs, learning_rate, gd_tech)

        predictTestSample(all_test_results, test_sample, n, nn)

    return all_test_results


def SCV_v2(all_pos_train_sample, all_neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate, gd_tech):
    avg_accuracy_list = []

    each_fold_size = int(float(total_train_size) / num_folds + 0.5)
    pos_ratio = float(len(all_pos_train_sample)) / total_train_size
    each_fold_pos_size = int(float(each_fold_size) * pos_ratio + 0.5)
    each_fold_neg_size = each_fold_size - each_fold_pos_size

    #random.shuffle(all_pos_train_sample)
    #random.shuffle(all_neg_train_sample)

    # Initialize all weights of all perceptrons in neural network to [-1, 1)
    nn_list = []
    for n in range(num_folds):
        random.seed(num_epochs)
        np.random.seed(num_epochs)
        nn = NeuralNetwork(len(header) - 1, learning_rate, gd_tech)
        nn_list.append(nn)

    for i in range(num_epochs):
        time_start = time.clock()

        all_test_results = [None] * (total_train_size)

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
            #random.shuffle(test_sample)

            pos_train_sample = all_pos_train_sample[0 : pos_test_start_idx] + all_pos_train_sample[(pos_test_end_idx + 1) : ]
            neg_train_sample = all_neg_train_sample[0 : neg_test_start_idx] + all_neg_train_sample[(neg_test_end_idx + 1) : ]

            train_sample = pos_train_sample + neg_train_sample;
            #random.shuffle(train_sample)

            random.shuffle(train_sample)
            for t in train_sample:
                # Input training sample to the network and compute all perceptrons' outputs
                nn_list[n].calOutputs(t)

                # Update weights based on gradient descent theory
                nn_list[n].updateWeight(t)

            predictTestSample(all_test_results, test_sample, n, nn_list[n])

        time_elapsed = (time.clock() - time_start)
        print('i = {}, time_elapsed = {}'.format(i, time_elapsed))

        match_num = 0;
        for a in all_test_results:
            if (a[1] == a[2]):
                match_num += 1

        print('total_train_size = {}, match_num = {}, accuracy = {}'.format(total_train_size, match_num, float(match_num) / total_train_size))
        avg_accuracy_list.append(float(match_num) / total_train_size)

    return avg_accuracy_list

def plotPartB1(pos_train_sample, neg_train_sample, total_train_size, gd_tech):
    #epoch_list = [25, 50, 75, 100]
    epoch_list = [1, 2, 3, 4]
    #epoch_list = [5, 15, 20, 25]
    #epoch_list = [10, 20, 30, 40]
    #epoch_list = [1]
    #epoch_list = [(i + 1) for i in range(10)]
    avg_accuracy_list = []

    for e in epoch_list:
        time_start = time.clock()
        match_num = 0;
        all_test_results = SCV(pos_train_sample, neg_train_sample, total_train_size, 10, e, 0.01, gd_tech)

        time_elapsed = (time.clock() - time_start)
        print('e = {}, time_elapsed = {}'.format(e, time_elapsed))

        for a in all_test_results:
            if (a[1] == a[2]):
                match_num += 1

        print('total_train_size = {}, match_num = {}, accuracy = {}'.format(total_train_size, match_num, float(match_num) / total_train_size))
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
    plt.title("Part B 1 - {}".format(gd_tech))
    l = plt.legend(loc = 4)
    plt.show()

def plotPartB2(pos_train_sample, neg_train_sample, total_train_size, gd_tech):
    num_folds_list = [5, 10, 15, 20, 25]
    avg_accuracy_list = []

    for n in num_folds_list:
        match_num = 0;
        all_test_results = SCV(pos_train_sample, neg_train_sample, total_train_size, n, 50, 0.1, gd_tech)

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
    plt.title("Part B 2 - {}".format(gd_tech))
    l = plt.legend(loc = 4)
    plt.show()

def plotPartB3(pos_train_sample, neg_train_sample, total_train_size, gd_tech):
    all_test_results = SCV(pos_train_sample, neg_train_sample, total_train_size, 10, 50, 0.1, gd_tech)
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
    plt.title("Part B 3 - {}".format(gd_tech))
    plt.show()

def plotPartB4(pos_train_sample, neg_train_sample, total_train_size, gd_tech):
    num_epochs = 10
    epoch_list = [(i + 1) for i in range(num_epochs)]

    avg_accuracy_list = SCV_v2(pos_train_sample, neg_train_sample, total_train_size, 10, num_epochs, 0.01, gd_tech)

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
    plt.title("Part B 4 - {}".format(gd_tech))
    l = plt.legend(loc = 4)
    plt.show()

def reindex(start, pos_train_sample, neg_train_sample):
    i = start
    for pos in pos_train_sample:
        pos[-2] = i
        i += 1
    for neg in neg_train_sample:
        neg[-2] = i
        i += 1
    return i

def plotPartB5(pos_train_sample, neg_train_sample, gd_tech):
    num_size = 10
    upper_size = 2000
    datasize_list = [upper_size/num_size*(i+1) for i in range(num_size)]
    random.shuffle(pos_train_sample)
    random.shuffle(neg_train_sample)

    start_idx = 0
    start_pos_idx = 0
    start_neg_idx = 0
    gd_tech_list = ["SGD", "AdaGrad", "Adam"]
    avg_accuracy_dict = dict()
    for gd_tech in gd_tech_list:
        avg_accuracy_dict[gd_tech] = []
    for datasize in datasize_list:
        pos_size = int(round(datasize/2.0)) if datasize < upper_size else len(pos_train_sample)
        neg_size = datasize - pos_size if datasize < upper_size else len(neg_train_sample)
        pos_sample = pos_train_sample[:pos_size]
        neg_sample = neg_train_sample[:neg_size]
        start_idx = reindex(start_idx, pos_sample[start_pos_idx:pos_size], neg_sample[start_neg_idx:neg_size])
        start_pos_idx = pos_size
        start_neg_idx = neg_size
        print len(pos_sample), len(neg_sample), datasize
        for gd_tech in gd_tech_list:
            match_num = 0
            time_start = time.clock()
            all_test_results = SCV(pos_sample, neg_sample, datasize, 2, 2, 0.01, gd_tech)
            for a in all_test_results:
                if (a[1] == a[2]):
                    match_num += 1
            accuracy = float(match_num) / datasize
            time_elapsed = (time.clock() - time_start)
            print('{} total_train_size = {}, match_num = {}, accuracy = {}, time_elapsed = {}'.format(gd_tech, datasize, match_num, accuracy, time_elapsed))
            #avg_accuracy_list.append(accuracy)
            avg_accuracy_dict[gd_tech].append(accuracy)

    for gd_tech, color in zip(gd_tech_list, ['r', 'g', 'b']):
        new_x, new_y = zip(*sorted(zip(datasize_list, [i * 100 for i in avg_accuracy_dict[gd_tech]])))
        plt.plot(new_x, new_y, 'go-', label=gd_tech, color=color)

    global_min = max(min([min([i * 100 for i in avg_accuracy_dict[gd_tech]]) for gd_tech in gd_tech_list]) - 1, 0)
    global_max = min(max([max([i * 100 for i in avg_accuracy_dict[gd_tech]]) for gd_tech in gd_tech_list]) + 1, 100)
    plt.axis([1, datasize_list[-1], global_min, global_max])
    plt.xticks(new_x)
    #plt.yticks(new_y)
    plt.grid(which='both')
    plt.xlabel('data size')
    plt.ylabel('% of accuracy')
    plt.title("Accuracy vs Data Size")
    l = plt.legend(loc = 4)
    plt.show()

def plot_loss_epoch(pos_train_sample, neg_train_sample, total_train_size, num_epochs):
    #num_epochs = 10
    epoch_list = [(i + 1) for i in range(num_epochs)]

    avg_loss_list = SCV_v3(pos_train_sample, neg_train_sample, total_train_size, 10, num_epochs, 0.01, 'SGD')
    new_x, new_y = zip(*sorted(zip(epoch_list, avg_loss_list)))
    plt.plot(new_x, new_y, 'ro-', label='SGD')

    avg_loss_list_adagrad = SCV_v3(pos_train_sample, neg_train_sample, total_train_size, 10, num_epochs, 0.01, 'AdaGrad')
    new_x, new_y = zip(*sorted(zip(epoch_list, avg_loss_list_adagrad)))
    plt.plot(new_x, new_y, 'go-', label='AdaGrad')

    avg_loss_list_adam = SCV_v3(pos_train_sample, neg_train_sample, total_train_size, 10, num_epochs, 0.01, 'Adam')
    new_x, new_y = zip(*sorted(zip(epoch_list, avg_loss_list_adam)))
    plt.plot(new_x, new_y, 'bo-', label='Adam')

    global_min = min([ min(avg_loss_list), min(avg_loss_list_adagrad), min(avg_loss_list_adam) ])
    global_max = max([ max(avg_loss_list), max(avg_loss_list_adagrad), max(avg_loss_list_adam) ])
    plt.axis([1, epoch_list[-1], 0, global_max + 0.1])
    plt.grid(which='both')
    plt.xlabel('# of epoch')
    plt.ylabel('cross entropy loss')
    plt.title("Cross Entropy Loss vs Epoch")
    l = plt.legend(loc = 4)
    #plt.show()
    plt.savefig('loss_vs_epoch', format='png')

def main():
    #assert len(sys.argv) == 5

    # Part A - Programming
    global header
    #header, train_sample = parseARFF(sys.argv[1])
    header, train_sample = parseMNIST_target(sys.argv[1], 3, 2000)

    #num_folds = int(sys.argv[2])
    #learning_rate = float(sys.argv[3])
    #num_epochs = int(sys.argv[4])

    # Label training samples for outputing prediction results in the original order
    for i in range(len(train_sample)):
        train_sample[i].append(train_sample[i][-1])
        train_sample[i][-2] = i

    total_train_size = len(train_sample)
    pos_train_sample, neg_train_sample = separateTrainSample(header, train_sample)

    gd_tech = 'AdaGrad' # 'SGD' or 'AdaGrad' or 'Adam'

    # ************************************************
    # **** This part is for customized parameters ****
    '''all_test_results = SCV(pos_train_sample, neg_train_sample, total_train_size, num_folds, num_epochs, learning_rate, gd_tech)

    # Print outputs in the following order: fold_of_instance predicted_class actual_class confidence_of_prediction
    for a in all_test_results:
        print("{} {} {} {}".format(a[0], a[1], a[2], a[3]))'''
    # **** Endo of This part is for customized parameters ****
    # ************************************************

    # Part B 1 - Programming: Accuracy vs Epoch
    #plotPartB1(pos_train_sample, neg_train_sample, total_train_size, gd_tech)

    # Part B 2 - Programming: Accuracy vs Fold Num
    #plotPartB2(pos_train_sample, neg_train_sample, total_train_size, gd_tech)

    # Part B 3 - Programming: ROC curve
    #plotPartB3(pos_train_sample, neg_train_sample, total_train_size, gd_tech)

    # Part B 4 - Output accuracy for each epoch
    #plotPartB4(pos_train_sample, neg_train_sample, total_train_size, gd_tech)


    # Part B 5 - Output accuracy for each data size
    #plotPartB5(pos_train_sample, neg_train_sample, gd_tech)
    plot_loss_epoch(pos_train_sample, neg_train_sample, total_train_size, 10)

if __name__ == "__main__":
    main()
