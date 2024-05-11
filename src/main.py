import time
import warnings
import csv

from ucimlrepo import fetch_ucirepo
from algorithm.optimization_arm import OptimizationARM
from algorithm.classic_arm import ClassicARM
from niapy.algorithms.basic import HarrisHawksOptimization, BatAlgorithm, SineCosineAlgorithm, GreyWolfOptimizer, \
    MothFlameOptimizer, DifferentialEvolution, ParticleSwarmOptimization, ArtificialBeeColonyAlgorithm, \
    FishSchoolSearch, MonarchButterflyOptimization
from preprocessing.ucimlrepo import *
from src.algorithm.aerial.aerial import Aerial
from src.algorithm.arm_ae.armae import ARMAE
from src.util.file import *
from src.util.rule_quality import *
from multiprocessing.pool import Pool

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

    datasets += [breast_cancer]
    print("Following datasets are loaded:", [dataset.metadata.name for dataset in datasets])

    return datasets


def execute(parameters):
    """
    a single execution of the tested algorithms
    :return:
    """
    dataset, algorithm = parameters
    opt_stats, opt_rules = algorithm.learn_rules(dataset)
    return opt_stats, opt_rules


if __name__ == '__main__':
    dataset_list = get_datasets()
    results = {}
    params = get_algorithm_parameters()

    fpgrowth = ClassicARM(min_support=params['MIN_SUPPORT'], min_confidence=params['MIN_CONFIDENCE'],
                          algorithm="fpgrowth")
    hmine = ClassicARM(min_support=params['MIN_SUPPORT'], min_confidence=params['MIN_CONFIDENCE'], algorithm="hmine")

    aerial = Aerial(noise_factor=params["NOISE_FACTOR"], max_antecedents=params["NUMBER_OF_ANTECEDENTS"],
                    similarity_threshold=params["SIMILARITY_THRESHOLD"])

    for dataset in dataset_list:
        print("Learning rules from dataset:", dataset.metadata.name, "...")
        classical_arm_input = prepare_classic_arm_input(dataset.data.features)
        optimization_based_arm_input = prepare_opt_arm_input(dataset)

        results[dataset.metadata.name] = {}
        results[dataset.metadata.name]["bat"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["gwo"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["sc"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fss"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fpgrowth"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["hmine"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["arm-ae"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["aerial"] = {"stats": [], "rules": None}

        # classical ARM
        fpgrowth_stats, fpgrowth_rules = fpgrowth.mine_rules(classical_arm_input)
        if fpgrowth_stats:
            results[dataset.metadata.name]["fpgrowth"]["stats"].append(fpgrowth_stats)
            results[dataset.metadata.name]["fpgrowth"]["rules"] = fpgrowth_rules

        hmine_stats, hmine_rules = hmine.mine_rules(classical_arm_input)
        if hmine_stats:
            results[dataset.metadata.name]["hmine"]["stats"].append(hmine_stats)
            results[dataset.metadata.name]["hmine"]["rules"] = hmine_rules

        # optimization-based ARM

        # Run all 5 optimization-based ARM algorithms in parallel. Running one algorithm 'NUM_OF_RUNS' times in
        # parallel results in the same rules, therefore, we parallelize running different algorithms at the same
        # time. Also, we need to re-create algorithm instances to avoid having the same results in each run
        pool = Pool(12)
        tasks = []
        for index in range(params['NUM_OF_RUNS']):
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(SineCosineAlgorithm(params['INIT_POPULATION_SIZE']),
                                                               max_evals=params['MAX_EVALS'])))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(GreyWolfOptimizer(params['INIT_POPULATION_SIZE']),
                                                               max_evals=params['MAX_EVALS'])))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(BatAlgorithm(params['INIT_POPULATION_SIZE']),
                                                               max_evals=params['MAX_EVALS'])))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(FishSchoolSearch(params['INIT_POPULATION_SIZE']),
                                                               max_evals=params['MAX_EVALS'])))
        #
        stats = pool.map(execute, tasks)
        for index in range(len(stats)):
            if stats[index][0]:
                if index % 4 == 0:
                    results[dataset.metadata.name]["sc"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["sc"]["rules"] = stats[index][1]
                if index % 4 == 1:
                    results[dataset.metadata.name]["gwo"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["gwo"]["rules"] = stats[index][1]
                if index % 4 == 2:
                    results[dataset.metadata.name]["bat"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["bat"]["rules"] = stats[index][1]
                if index % 4 == 3:
                    results[dataset.metadata.name]["fss"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["fss"]["rules"] = stats[index][1]
        #
        pool.close()
        pool.join()

        # Aerial (2024)
        aerial.create_input_vectors(dataset)
        aerial_training_time = aerial.train()
        aerial_association_rules, ae_exec_time = aerial.generate_rules()
        aerial_stats = aerial.calculate_stats(aerial_association_rules, classical_arm_input,
                                              ae_exec_time + aerial_training_time)
        if aerial_stats:
            results[dataset.metadata.name]["aerial"]["stats"].append(aerial_stats)
            results[dataset.metadata.name]["aerial"]["rules"] = aerial_association_rules

        # # ARM-AE from Berteloot et al. (2023)
        one_hot_encoded = one_hot_encoding(classical_arm_input)
        arm_ae = ARMAE(len(one_hot_encoded.loc[0]), maxEpoch=int(params["EPOCHS"]),
                       batchSize=int(params["BATCH_SIZE"]), learningRate=float(params["LEARNING_RATE"]),
                       likeness=float(params["LIKENESS"]))
        dataLoader = arm_ae.dataPreprocessing(one_hot_encoded)
        arm_ae_training_time = arm_ae.train(dataLoader)
        arm_ae.generateRules(one_hot_encoded,
                             numberOfRules=int(len(aerial_association_rules) / one_hot_encoded.shape[1]),
                             nbAntecedent=params["NUMBER_OF_ANTECEDENTS"])
        arm_ae_stats, arm_ae_rules = arm_ae.reformat_rules(classical_arm_input, list(one_hot_encoded.columns))
        if arm_ae_stats:
            results[dataset.metadata.name]["arm-ae"]["stats"].append(arm_ae_stats)
            results[dataset.metadata.name]["arm-ae"]["rules"] = arm_ae_rules

    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for dataset in results:
            writer.writerow(["Dataset: " + dataset])
            writer.writerow(
                ["Algorithm", "Rule Count", "Execution Time", "Support", "Confidence", "Lift", "Zhang", "Coverage",
                 "Interestingness", "Yules'Q"])
            for algorithm in results[dataset]:
                if results[dataset][algorithm]['stats'] and len(results[dataset][algorithm]['stats']) > 0:
                    row = print_stats(algorithm, max(results[dataset][algorithm]["stats"], key=lambda x: x[3]))
                else:
                    row = [algorithm, "No rules found!"]
                writer.writerow(row)

    calculate_rule_overlap(results)
