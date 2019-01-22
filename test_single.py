from Dataset.DatasetReader import DatasetReader
from Utilities.Support import evaluate_algorithm
import os
import traceback

from Recommenders.ContentBasedRecommender import ContentBasedRecommender
from Recommenders.RP3BetaRecommender import RP3betaRecommender

if __name__ == '__main__':

    dataReader = DatasetReader(modelNumber="1")

    URM = dataReader.get_URM()
    ICM = dataReader.get_ICM()

    URM_train = dataReader.get_URM_train()
    URM_test = dataReader.get_URM_test()

    target_playlists = dataReader.get_target_playlists()
    output_root_path = "results/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "results.txt", "a")

    recommender_list = [
        RP3betaRecommender
    ]

    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))

            if recommender_class == ContentBasedRecommender:
                recommender = recommender_class(URM_train, ICM)
            else:
                recommender = recommender_class(URM_train)

            recommender.fit()
            results_run = evaluate_algorithm(URM_test, target_playlists, recommender, 10)
            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
