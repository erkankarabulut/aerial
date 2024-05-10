from niaarm import get_rules, Dataset
from niapy.algorithms.basic import HarrisHawksOptimization
import pandas as pd
import numpy as np


class OptimizationARM:
    """
    An implementation of single-objective optimization-based ARM algorithms using NiaARM and NiaPy packages
    """

    def __init__(self, algorithm, max_evals=50000):
        self.max_evals = max_evals
        self.algorithm = algorithm

        self.metrics = ['support', 'confidence']

    def learn_rules(self, transactions):
        dataset = Dataset(transactions)
        rules, run_time = get_rules(dataset, algorithm=self.algorithm, metrics=self.metrics,
                                    max_evals=self.max_evals, logging=False)
        if len(rules) == 0:
            return False
        coverage = self.calculate_coverage(rules, transactions)
        support, confidence, lift, zhangs, interestingness, yulesq = \
            rules.mean("support"), rules.mean("confidence"), rules.mean("lift"), rules.mean("zhang"), \
                rules.mean("interestingness"), rules.mean("yulesq")
        # rules = self.reformat_rules(rules)
        return [len(rules), run_time, support, confidence, lift, zhangs, coverage, interestingness, yulesq], rules

    @staticmethod
    def calculate_coverage(rules, dataset):
        rule_coverage = np.zeros(len(dataset))

        for index, row in dataset.iterrows():
            for rule in rules:
                covered = True
                for item in rule.antecedent:
                    if item.categories:
                        if item.categories[0] not in list(row):
                            covered = False
                            break
                    else:
                        covered = False
                        for key, value in row.items():
                            if item.name == key:
                                if item.min_val <= value <= item.max_val:
                                    covered = True
                                    break
                if covered:
                    rule_coverage[index] = 1
                    break

        return sum(rule_coverage) / len(dataset)

    def reformat_rules(self, rules):
        reformatted_rules = []
        for rule in rules:
            print(rule)
        return reformatted_rules
