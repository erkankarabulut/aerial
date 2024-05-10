# this script is taken from: https://github.com/TheophileBERTELOOT/ARM-AE/blob/master/ARMAE.py
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from src.algorithm.arm_ae.autoencoder import AutoEncoder
from src.util.rule_quality import *

import copy
import time


class ARMAE:
    def __init__(self, dataSize, learningRate=1e-3, maxEpoch=5,
                 batchSize=128, hiddenSize='dataSize', likeness=0.4, columns=[], isLoadedModel=False,
                 IM=['support', 'confidence']):
        self.arm_ae_training_time = 0
        self.exec_time = None
        self.dataSize = dataSize
        self.learningRate = learningRate
        self.likeness = likeness
        self.IM = IM
        self.hiddenSize = hiddenSize
        self.isLoadedModel = isLoadedModel
        self.columns = columns
        if self.hiddenSize == 'dataSize':
            self.hiddenSize = self.dataSize
        self.maxEpoch = maxEpoch
        self.x = []
        self.y_ = []
        self.batchSize = batchSize
        self.model = AutoEncoder(self.dataSize)
        self.criterion = L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learningRate)

        self.results = []

    def dataPreprocessing(self, d):
        self.columns = d.columns
        trainTensor = torch.tensor(d.values)
        dataLoader = DataLoader(trainTensor.float(), batch_size=self.batchSize, shuffle=True)
        x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
        torch.nan_to_num(x, nan=0.0, posinf=0.0)
        return dataLoader

    def save(self, p):
        self.model.save(p)

    def load(self, encoderPath, decoderPath):
        self.model.load(encoderPath, decoderPath)

    def train(self, dataLoader):
        armae_training_start = time.time()
        for epoch in range(self.maxEpoch):
            for data in dataLoader:
                d = Variable(data)
                output = self.model.forward(d)
                self.y_ = output[0]
                self.x = d
                loss = self.criterion(output[0], d)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            # print('epoch [{}/{}], loss:{:.4f}'
            #       .format(epoch + 1, self.maxEpoch, loss.data))
        self.arm_ae_training_time = time.time() - armae_training_start

    def computeMeasures(self, antecedent, consequent, data):
        measures = []
        if 'support' in self.IM:
            rules = copy.deepcopy(antecedent)
            rules.append(consequent)
            PAC = data[data.columns[rules]]
            PAC = np.sum(PAC, axis=1)
            PAC = PAC == len(rules)
            PAC = np.sum(PAC)
            PAC = PAC / len(data)
            measures.append(round(PAC, 2))
        if 'confidence' in self.IM:
            PA = data[data.columns[antecedent]]
            PA = np.sum(PA, axis=1)
            PA = PA == len(antecedent)
            PA = np.sum(PA)
            PA = PA / len(data)
            if PA != 0:
                conf = PAC / PA
            else:
                conf = 0
            measures.append(round(conf, 2))
        return measures

    def computeSimilarity(self, allAntecedents, antecedentsArray, nbantecedent):
        onlySameSize = [x for x in allAntecedents if len(x) >= len(antecedentsArray)]
        maxSimilarity = 0
        for antecedentIndex in range(len(onlySameSize)):
            antecedents = onlySameSize[antecedentIndex]
            similarity = 0
            for item in antecedents:
                if item in antecedentsArray:
                    similarity += 1
            similarity /= nbantecedent
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        return maxSimilarity

    def generateRules(self, data, numberOfRules=2, nbAntecedent=2):
        # print('begin rules generation')
        timeCreatingRule = 0
        timeComputingMeasure = 0

        for consequent in range(self.dataSize):
            allAntecedents = []
            for j in range(numberOfRules):
                antecedentsArray = []
                for i in range(nbAntecedent):
                    t1 = time.time()
                    consequentArray = np.zeros(self.dataSize)
                    consequentArray[consequent] = 1
                    consequentArray[antecedentsArray] = 1
                    consequentArray = torch.tensor(consequentArray)
                    consequentArray = consequentArray.unsqueeze(0)
                    output = self.model(consequentArray.float())
                    output = output.cpu()
                    output = np.array(output.detach().numpy())
                    output = pd.DataFrame(output.reshape(self.dataSize, -1))
                    potentialAntecedentsArray = output[0].nlargest(len(data.loc[0]))
                    for antecedent in potentialAntecedentsArray.keys():
                        potentialAntecedents = copy.deepcopy(antecedentsArray)
                        potentialAntecedents.append(antecedent)
                        potentialAntecedents = sorted(potentialAntecedents)
                        if antecedent != consequent and antecedent not in antecedentsArray and self.computeSimilarity(
                                allAntecedents, potentialAntecedents, nbAntecedent) <= self.likeness:
                            antecedentsArray.append(antecedent)
                            break
                    t2 = time.time()
                    measures = self.computeMeasures(copy.deepcopy(antecedentsArray), consequent, data)
                    t3 = time.time()
                    ruleProperties = [list(sorted(copy.deepcopy(antecedentsArray))), [consequent]]
                    ruleProperties += measures
                    self.results.append(ruleProperties)
                    allAntecedents.append(sorted(copy.deepcopy(antecedentsArray)))
                    timeCreatingRule += t2 - t1
                    timeComputingMeasure += t3 - t2

        self.exec_time = timeCreatingRule + self.arm_ae_training_time

    def reformat_rules(self, data, data_columns):
        """
        Karabulut:
        ARM-AE generates rules in the form of indexes, e.g. feature1 implies feature2
        This method added to ARM-AE code to reformat the rules and use feature names instead
        In addition, it also calculates other rule quality measures besides support and confidence
        :param data_columns:
        :param data: non-formatted data (not one-hot encoded)
        :return:
        """
        formatted_rules = []
        rule_coverage = np.zeros(len(data))
        for rule in self.results:
            # ignore the rules with 0 support value (as in the original paper)
            if rule[2] == 0:
                continue
            rule_stats = {}
            # replace feature indexes in antecedent
            for index in range(len(rule[0])):
                rule[0][index] = data_columns[rule[0][index]]
            # replace feature indexes in consequent
            for index in range(len(rule[1])):
                rule[1][index] = data_columns[rule[1][index]]
            rule_stats = {"antecedents": rule[0], "consequent": rule[1], "support": rule[2], "confidence": rule[3]}

            # calculate rule quality criteria besides support and confidence
            ant_count = 0
            cons_count = 0
            ant_and_cons_count = 0
            cons_no_ant_count = 0
            ant_no_cons_count = 0
            no_ant_no_cons_count = 0
            for transaction_index in range(len(data)):
                transaction = data[transaction_index]
                # count the occurrences of antecedent and consequent side of the rule in each transaction, bot separate
                # co-occurrences
                if all(item in transaction for item in rule_stats['antecedents']):
                    ant_count += 1
                    rule_coverage[transaction_index] = 1
                    if all(item in transaction for item in rule_stats['consequent']):
                        ant_and_cons_count += 1
                        cons_count += 1
                    else:
                        ant_no_cons_count += 1
                elif all(item in transaction for item in rule_stats['consequent']):
                    cons_no_ant_count += 1
                    cons_count += 1
                else:
                    no_ant_no_cons_count += 1

            rule_stats["lift"] = calculate_lift(rule_stats["support"], rule_stats["confidence"])
            # rule_stats["conviction"] = calculate_conviction(rule_stats["support"], rule_stats["confidence"])
            rule_stats["zhangs_metric"] = calculate_zhangs_metric(rule_stats["support"], (ant_count / len(data)),
                                                                  (cons_count / len(data)))
            rule_stats["interestingness"] = calculate_interestingness(rule_stats['confidence'], rule_stats['support'],
                                                                      (cons_count / len(data)), len(data))
            rule_stats["yulesq"] = calculate_yulesq(ant_and_cons_count, no_ant_no_cons_count, cons_no_ant_count,
                                                    ant_no_cons_count)
            formatted_rules.append(rule_stats)

        stats = calculate_average_rule_quality(formatted_rules)
        stats["coverage"] = sum(rule_coverage) / len(data)
        return [len(formatted_rules), self.exec_time, stats['support'], stats["confidence"], stats["lift"],
                stats["zhangs_metric"], stats["coverage"], stats["interestingness"], stats["yulesq"]]
