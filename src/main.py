import warnings
import csv

import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo
from algorithm.optimization_arm import OptimizationARM
from algorithm.classic_arm import ClassicARM
from niapy.algorithms.basic import HarrisHawksOptimization, BatAlgorithm, SineCosineAlgorithm, GreyWolfOptimizer, \
    MothFlameOptimizer
from preprocessing.ucimlrepo import *
from src.algorithm.arm_ae.armae import ARMAE
from src.algorithm.ae_semrl.ae_semrl import AESemRL
from src.util.file import *
from src.util.rule_quality import *

# not a good practice, but to keep the console output clean for now
warnings.filterwarnings("ignore")


def print_stats(algorithm, stats_array):
    print("count,time,support,confidence,lift,zhang,coverage,interestingness,yulesq\n"
          "(%.f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)" %
          (stats_array[0], stats_array[1], stats_array[2], stats_array[3], stats_array[4], stats_array[5],
           stats_array[6], stats_array[7], stats_array[8]))
    return [algorithm, round(stats_array[0], 3), round(stats_array[1], 3), round(stats_array[2], 3),
            round(stats_array[3], 3), round(stats_array[4], 3), round(stats_array[5], 3), round(stats_array[6], 3),
            round(stats_array[7], 3), round(stats_array[8], 3)]


def get_datasets():
    datasets = []

    print("Loading the datasets ...")
    # hayes_roth = fetch_ucirepo(id=44)
    # solar_flare = fetch_ucirepo(id=89)
    breast_cancer = fetch_ucirepo(id=14)
    # congress_voting_records = fetch_ucirepo(id=105)
    # mushroom = fetch_ucirepo(id=73)
    # chess = fetch_ucirepo(id=22)
    # connect_4 = fetch_ucirepo(id=26)
    # molecular_biology = fetch_ucirepo(id=69)
    # us_census_data = fetch_ucirepo(id=116)

    datasets += [breast_cancer]
    print("Following datasets are loaded:", [dataset.metadata.name for dataset in datasets])

    return datasets


if __name__ == '__main__':
    dataset_list = get_datasets()
    results = {}
    params = get_algorithm_parameters()

    hho = OptimizationARM(HarrisHawksOptimization(params['INIT_POPULATION_SIZE']),
                          max_iterations=params['MAX_ITERATIONS'])
    bat = OptimizationARM(BatAlgorithm(params['INIT_POPULATION_SIZE']), max_iterations=params['MAX_ITERATIONS'])
    gwo = OptimizationARM(GreyWolfOptimizer(params['INIT_POPULATION_SIZE']), max_iterations=params['MAX_ITERATIONS'])
    mfo = OptimizationARM(MothFlameOptimizer(params['INIT_POPULATION_SIZE']), max_iterations=params['MAX_ITERATIONS'])
    sca = OptimizationARM(SineCosineAlgorithm(params['INIT_POPULATION_SIZE']), max_iterations=params['MAX_ITERATIONS'])

    fpgrowth = ClassicARM(min_support=params['MIN_SUPPORT'], min_confidence=params['MIN_CONFIDENCE'],
                          algorithm="fpgrowth")
    hmine = ClassicARM(min_support=params['MIN_SUPPORT'], min_confidence=params['MIN_CONFIDENCE'], algorithm="hmine")

    ae_semrl = AESemRL(noise_factor=params["NOISE_FACTOR"], max_antecedents=params["NUMBER_OF_ANTECEDENTS"],
                       similarity_threshold=params["SIMILARITY_THRESHOLD"])

    for dataset in dataset_list:
        print("Learning rules from dataset:", dataset.metadata.name, "...")
        classical_arm_input = prepare_classic_arm_input(dataset.data.features)
        optimization_based_arm_input = pd.DataFrame(classical_arm_input, columns=list(dataset.data.features.columns))

        results[dataset.metadata.name] = {}
        results[dataset.metadata.name]["hho"] = []
        results[dataset.metadata.name]["bat"] = []
        results[dataset.metadata.name]["gwo"] = []
        results[dataset.metadata.name]["mfo"] = []
        results[dataset.metadata.name]["sca"] = []
        results[dataset.metadata.name]["fpgrowth"] = []
        results[dataset.metadata.name]["hmine"] = []
        results[dataset.metadata.name]["arm-ae"] = []
        results[dataset.metadata.name]["ae-semrl"] = []

        # classical ARM
        # fpgrowth_stats = fpgrowth.mine_rules(classical_arm_input)
        # if fpgrowth_stats:
        #     results[dataset.metadata.name]["fpgrowth"].append(fpgrowth_stats)

        # hmine_stats = hmine.mine_rules(classical_arm_input)
        # if hmine_stats:
        #     results[dataset.metadata.name]["hmine"].append(hmine_stats)

        for i in range(params['NUM_OF_RUNS']):
            # optimization-based ARM
            # hho_stats = hho.learn_rules(optimization_based_arm_input)
            # if hho_stats:
            #     results[dataset.metadata.name]["hho"].append(hho_stats)

            # bat_stats = hho.learn_rules(optimization_based_arm_input)
            # if bat_stats:
            #     results[dataset.metadata.name]["bat"].append(bat_stats)

            # gwo_stats = gwo.learn_rules(optimization_based_arm_input)
            # if gwo_stats:
            #     results[dataset.metadata.name]["gwo"].append(gwo_stats)
#
            # mfo_stats = mfo.learn_rules(optimization_based_arm_input)
            # if mfo_stats:
            #     results[dataset.metadata.name]["mfo"].append(mfo_stats)
#
            # sca_stats = sca.learn_rules(optimization_based_arm_input)
            # if sca_stats:
            #     results[dataset.metadata.name]["sca"].append(sca_stats)
#
            # one_hot_encoded = one_hot_encoding(classical_arm_input)

            # AE-SemRL (2024)
            ae_semrl.create_input_vectors(dataset)
            ae_semrl.train()
            ae_semrl_association_rules, ae_exec_time = ae_semrl.generate_rules()
            ae_semrl_stats = ae_semrl.calculate_stats(ae_semrl_association_rules, classical_arm_input, ae_exec_time)
            if ae_semrl_stats:
                results[dataset.metadata.name]["ae-semrl"].append(ae_semrl_stats)

            # ARM-AE from Berteloot et al. (2023)
            # arm_ae = ARMAE(len(one_hot_encoded.loc[0]), maxEpoch=int(params["EPOCHS"]),
            #                batchSize=int(params["BATCH_SIZE"]), learningRate=float(params["LEARNING_RATE"]),
            #                likeness=float(params["LIKENESS"]))
            # dataLoader = arm_ae.dataPreprocessing(one_hot_encoded)
            # arm_ae.train(dataLoader)
            # arm_ae.generateRules(one_hot_encoded,
            #                      numberOfRules=int(len(ae_semrl_association_rules) / one_hot_encoded.shape[1]),
            #                      nbAntecedent=params["NUMBER_OF_ANTECEDENTS"])
            # arm_ae_stats = arm_ae.reformat_rules(classical_arm_input, list(one_hot_encoded.columns))
            # results[dataset.metadata.name]["arm-ae"].append(arm_ae_stats)

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for dataset in results:
            writer.writerow(["Dataset: " + dataset])
            writer.writerow(
                ["Algorithm", "Rule Count", "Execution Time", "Support", "Confidence", "Lift", "Zhang", "Coverage",
                 "Interestingness", "Yules'Q"])
            for algorithm in results[dataset]:
                if len(results[dataset][algorithm]) > 0:
                    row = print_stats(algorithm, pd.DataFrame(results[dataset][algorithm]).mean())
                else:
                    row = [algorithm, "No rules found!"]
                writer.writerow(row)
