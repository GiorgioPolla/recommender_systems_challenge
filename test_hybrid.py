from Dataset.DatasetReader import DatasetReader
from Recommenders.HybridRecommender import HybridRecommender
from Utilities.Support import evaluate_algorithm

import os

if __name__ == '__main__':

    dataReader = DatasetReader()

    URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()
    ICM = dataReader.get_ICM()
    target_playlists = dataReader.get_target_playlists()
    seq = dataReader.get_sequential_playlists()

    output_root_path = "results/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "hybrid_results_all.txt", "a")

    print("Hybrid recommender tuning started!")

    recommender = HybridRecommender(URM_train, ICM)

    recommender.fit(0.0922, 0.1996, 0.2374, 0.4319, 0.3500, 0.5619)
    results = evaluate_algorithm(URM_test, target_playlists, recommender, 10)

    logFile.write(
        "Result: {}\nWeights {} {} {} {} {}\n".format(results, 0.0922, 0.1996, 0.2374, 0.4319, 0.3500, 0.5619))
    logFile.flush()
