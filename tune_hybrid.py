from Utilities.EvaluatorForSkopt import EvaluatorForSkopt
from Recommenders.HybridRecommender import HybridRecommender
from Dataset.DatasetReader import DatasetReader
from skopt import gp_minimize
from skopt import forest_minimize

if __name__ == '__main__':
    dataReader = DatasetReader()

    URM_train = dataReader.get_URM_train()
    URM_test = dataReader.get_URM_test()
    URM_validation = dataReader.get_URM_validation()
    ICM = dataReader.get_ICM()

    target_playlists = dataReader.get_target_playlists()

    recommender = HybridRecommender(URM_train, ICM)

    evaluator = EvaluatorForSkopt(URM_test, URM_validation, target_playlists, recommender, at=10)

    res = gp_minimize(evaluator.evaluate_algorithm,
                      [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], n_calls=35,
                      n_random_starts=7, verbose=True,
                      x0=[[0.0922, 0.1996, 0.2374, 0.4319, 0.3500, 0.5619],
                          [0.3000, 0.1000, 0.1000, 0.3000, 0.2000, 1.0000],
                          [0.1529, 0.3326, 0.7276, 0.7338, 0.9698, 0.9185]])

    print("Final result:")
    print("Best MAP: " + str(res.fun))
    print("Weights: " + str(res.x))
