import numpy as np
import pandas as pd
import time
from src.util.rule_quality import *
from src.preprocessing.ucimlrepo import *
from PAMI.frequentPattern.basic import ECLAT
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, hmine


class ClassicARM:
    def __init__(self, min_support, min_confidence, algorithm):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.algorithm = algorithm

    def mine_rules(self, dataset):
        one_hot_encoded_input = one_hot_encoding(dataset)
        print("One-hot encoded feature count:", len(list(one_hot_encoded_input.columns)))
        start = time.time()

        if self.algorithm == "fpgrowth":
            frq_items = fpgrowth(one_hot_encoded_input, self.min_support, use_colnames=True)
        else:
            frq_items = hmine(one_hot_encoded_input, self.min_support, use_colnames=True)

        if len(frq_items) == 0:
            return None

        rules = association_rules(frq_items, metric="confidence", min_threshold=self.min_confidence)
        exec_time = time.time() - start

        if len(rules) > 0:
            return self.calculate_stats(rules, exec_time, dataset)
        else:
            return None

    @staticmethod
    def calculate_stats(rules, exec_time, dataset):
        """
        The following rule quality criteria calculation is missing inside the MLxtend package:
        coverage, interestingness and Yules' Q.
        :param exec_time:
        :param rules:
        :param dataset:
        :return:
        """
        rule_stats = []
        rule_coverage = np.zeros(len(dataset))
        for index, row in rules.iterrows():
            ant_count = 0
            ant_and_cons_count = 0
            cons_no_ant_count = 0
            ant_no_cons_count = 0
            no_ant_no_cons_count = 0
            for transaction_index in range(len(dataset)):
                transaction = dataset[transaction_index]
                # count the occurrences of antecedent and consequent side of the rule in each transaction, bot separate
                # co-occurrences
                if all(item in transaction for item in row['antecedents']):
                    ant_count += 1
                    rule_coverage[transaction_index] = 1
                    if all(item in transaction for item in row['consequents']):
                        ant_and_cons_count += 1
                    else:
                        ant_no_cons_count += 1
                elif all(item in transaction for item in row['consequents']):
                    cons_no_ant_count += 1
                else:
                    no_ant_no_cons_count += 1

            row["interestingness"] = calculate_interestingness(row['confidence'], row['support'],
                                                               row['consequent support'], len(dataset))
            row["yulesq"] = calculate_yulesq(ant_and_cons_count, no_ant_no_cons_count, cons_no_ant_count,
                                             ant_no_cons_count)
            rule_stats.append(row)

        stats = calculate_average_rule_quality(rule_stats)
        stats["coverage"] = sum(rule_coverage) / len(dataset)
        return [len(rules), exec_time, stats['support'], stats["confidence"], stats["lift"], stats["zhangs_metric"],
                stats["coverage"], stats["interestingness"], stats["yulesq"]]
