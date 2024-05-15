# Aerial source code is inspired from AE SemRL: https://github.com/DiTEC-project/semantic-association-rule-learning
import random
import time

import torch
import copy

from itertools import chain, combinations
from torch import nn
from src.algorithm.aerial.autoencoder import AutoEncoder
from src.preprocessing.ucimlrepo import *
from src.util.rule_quality import *
import numpy as np


class Aerial:
    """
    AutoEncoder-based Association Rule Learning (Aerial) implementation
    """

    def __init__(self, noise_factor=0.5, similarity_threshold=0.8, max_antecedents=2):
        """
        @param noise_factor: amount of noise introduced for the one-hot encoded input of denoising Autoencoder
        @param similarity_threshold: feature similarity threshold
        @param max_antecedents: maximum number of antecedents that the learned rules will have
        """
        self.noise_factor = noise_factor
        self.similarity_threshold = similarity_threshold
        self.max_antecedents = max_antecedents

        self.model = None
        self.input_vectors = None
        self.softmax = nn.Softmax(dim=0)

    def create_input_vectors(self, dataset):
        """
        create input vectors for training the Autoencoder in the one-hot encoded form
        :param dataset:
        :return:
        """
        transactions = dataset.data.features
        columns = list(transactions.columns)

        # get input vectors in the form of one-hot encoded vectors
        unique_values, value_count = get_unique_values_per_column(dataset)
        vector_tracker = []
        feature_value_indices = []
        start = 0
        # track what each value in the input vector corresponds to
        # track where do values for each feature starts and end in the input feature
        for feature in unique_values:
            feature_value_indices.append({'start': start, 'end': start + len(unique_values[feature])})
            start += len(unique_values[feature])
            for value in unique_values[feature]:
                vector_tracker.append(feature + "__" + str(value))

        vector_list = []
        for transaction_index, transaction in transactions.iterrows():
            vector = list(np.zeros(value_count))
            for index in range(len(list(transaction))):
                vector[vector_tracker.index(columns[index] + "__" + str(transaction[index]))] = 1
            vector_list.append(vector)

        self.input_vectors = {
            "vector_list": vector_list,
            "vector_tracker_list": vector_tracker,
            "feature_value_indices": feature_value_indices
        }

    def generate_rules(self):
        """
        generate rules using the AE ARL algorithm
        """
        association_rules = []
        input_vector_size = len(self.input_vectors['vector_tracker_list'])

        start = time.time()
        powerset = list(chain.from_iterable(
            combinations(self.input_vectors["feature_value_indices"], r) for r in range(self.max_antecedents + 1)))
        copied_input_vector_list = copy.deepcopy(self.input_vectors['vector_list'])
        unique_input_vector_list = [list(x) for x in set(tuple(x) for x in copied_input_vector_list)]

        for category_list in powerset[1:]:
            unmarked_features = self.initialize_input_vectors(
                input_vector_size, self.input_vectors["feature_value_indices"], category_list)

            test_vectors = self.mark_features(unmarked_features, list(category_list), unique_input_vector_list)

            for test_vector in test_vectors:
                implication_probabilities = self.model(torch.FloatTensor(test_vector), self.input_vectors[
                    "feature_value_indices"]).detach().numpy().tolist()
                consequent_list = []
                for prob_index in range(len(implication_probabilities)):
                    self_implication = False
                    for temp_cat in category_list:
                        if temp_cat['start'] <= prob_index < temp_cat['end']:
                            self_implication = True
                            break
                    if self_implication:
                        continue
                    if implication_probabilities[prob_index] >= self.similarity_threshold:
                        consequent_list.append(prob_index)
                if len(consequent_list) > 0:
                    antecedent_list = [index for index in range(len(test_vector)) if test_vector[index] == 1]
                    new_rule = self.get_rule(antecedent_list, consequent_list)

                    for consequent in new_rule['consequents']:
                        association_rules.append({'antecedents': new_rule['antecedents'], 'consequent': consequent})

        execution_time = time.time() - start
        return association_rules, execution_time

    def calculate_stats(self, rules, transactions, exec_time):
        """
        calculate rule quality stats for the given set of rules based on the input transactions
        """
        rule_coverage = np.zeros(len(transactions))
        for rule_index in range(len(rules)):
            rule = rules[rule_index]
            ant_count = 0
            cons_count = 0
            co_occurrence_count = 0
            only_antecedence_occurrence_count = 0
            only_consequence_occurrence_count = 0
            no_ant_no_cons_count = 0
            for index in range(len(self.input_vectors['vector_list'])):
                encoded_transaction = self.input_vectors['vector_list'][index]
                antecedent_match = True
                for antecedent in rule['antecedents']:
                    if encoded_transaction[self.input_vectors['vector_tracker_list'].index(antecedent)] == 0:
                        antecedent_match = False
                        break
                if antecedent_match:
                    ant_count += 1
                    rule_coverage[index] = 1
                if encoded_transaction[self.input_vectors['vector_tracker_list'].index(rule['consequent'])] == 1:
                    cons_count += 1
                    if antecedent_match:
                        co_occurrence_count += 1
                    else:
                        only_consequence_occurrence_count += 1
                elif antecedent_match:
                    only_antecedence_occurrence_count += 1
                else:
                    no_ant_no_cons_count += 1

            num_transactions = len(transactions)
            support_body = ant_count / num_transactions
            support_head = cons_count / num_transactions

            rule['support'] = co_occurrence_count / num_transactions
            rule['confidence'] = rule['support'] / support_body
            rule['lift'] = rule['confidence'] / support_head
            rule['leverage'] = rule['support'] - (support_body * support_head)
            # rule["conviction"] = calculate_conviction(rule["support"], rule["confidence"])
            rule["zhangs_metric"] = calculate_zhangs_metric(rule["support"],
                                                            (ant_count / len(self.input_vectors['vector_list'])),
                                                            (cons_count / len(self.input_vectors['vector_list'])))
            rule["interestingness"] = calculate_interestingness(rule['confidence'], rule['support'],
                                                                (cons_count / len(self.input_vectors['vector_list'])),
                                                                len(self.input_vectors['vector_list']))
            rule["yulesq"] = calculate_yulesq(co_occurrence_count, no_ant_no_cons_count,
                                              only_consequence_occurrence_count,
                                              only_antecedence_occurrence_count)
            # just to make each rule in the same form, the other algorithms have the consequent as in a list form
            rule["consequent"] = [rule["consequent"]]

        if len(rules) == 0:
            return None
        stats = calculate_average_rule_quality(rules)
        stats["coverage"] = sum(rule_coverage) / len(self.input_vectors['vector_list'])
        return [len(rules), exec_time, stats['support'], stats["confidence"], stats["lift"],
                stats["zhangs_metric"], stats["coverage"], stats["interestingness"], stats["yulesq"]]

    @staticmethod
    def initialize_input_vectors(input_vector_size, categories, marked_categories) -> list:
        vector_with_unmarked_features = np.zeros(input_vector_size)
        for category in categories:
            if category not in marked_categories:
                vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                        category['end'] - category['start'])
        return list(vector_with_unmarked_features)

    def mark_features(self, unmarked_test_vector, categories, input_vectors, test_vectors=[]):
        if len(categories) == 0:
            return test_vectors
        category = categories.pop()
        new_test_vectors = []
        for i in range(category['end'] - category['start']):
            if len(test_vectors) > 0:
                for vector in test_vectors:
                    new_vector = copy.deepcopy(vector)
                    new_vector[category['start'] + i] = 1
                    indices = [i for i, x in enumerate(new_vector) if x == 1]
                    for input_vector in input_vectors:
                        total = sum([input_vector[index] for index in indices])
                        if total == len(indices):
                            new_test_vectors.append(new_vector)
                            break
            else:
                new_vector = copy.deepcopy(unmarked_test_vector)
                new_vector[category['start'] + i] = 1
                for input_vector in input_vectors:
                    if input_vector[category['start'] + i] == 1:
                        new_test_vectors.append(new_vector)
                        break
        return self.mark_features(unmarked_test_vector, categories, input_vectors, new_test_vectors)

    def get_rule(self, antecedents, consequents):
        rule = {'antecedents': [], 'consequents': []}
        for antecedent in antecedents:
            rule['antecedents'].append(self.input_vectors['vector_tracker_list'][antecedent])

        for consequent in consequents:
            rule['consequents'].append(self.input_vectors['vector_tracker_list'][consequent])

        return rule

    def train(self):
        """
        train the autoencoder
        """
        # pretrain categorical attributes from the knowledge graph, to create a numerical representation for them
        self.model = AutoEncoder(len(self.input_vectors['vector_list'][0]))

        # if not self.model.load("test"):
        training_time = self.train_ae_model()
        # self.model.save("test")
        return training_time

    def train_ae_model(self, loss_function=torch.nn.BCELoss(), lr=5e-3, epochs=5):
        """
        train the encoder on the given dataset
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=2e-8)
        vectors = self.input_vectors['vector_list']
        random.shuffle(vectors)

        training_start_time = time.time()
        for epoch in range(epochs):
            for index in range(len(vectors)):
                print("Training progress:", (index + 1 + (epoch * len(vectors))), "/", (len(vectors) * epochs),
                      end="\r")
                cat_vector = torch.FloatTensor(vectors[index])
                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)

                reconstructed = self.model(noisy_cat_vector, self.input_vectors["feature_value_indices"])

                partial_losses = []
                for category_range in self.input_vectors["feature_value_indices"]:
                    start = category_range['start']
                    end = category_range['end']
                    partial_losses.append(loss_function(reconstructed[start:end], cat_vector[start:end]))

                loss = sum(partial_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        training_time = time.time() - training_start_time
        return training_time
