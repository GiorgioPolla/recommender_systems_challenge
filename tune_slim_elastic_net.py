from Utilities.EvaluatorForSkopt import EvaluatorForSkopt
from Recommenders.SlimElasticNetRecommender import SlimElasticNetRecommender
from Dataset.DatasetReader import DatasetReader
from skopt import gp_minimize

if __name__ == '__main__':
    dataReader = DatasetReader()

    URM_train = dataReader.get_URM_train()
    URM_test = dataReader.get_URM_test()

    target_playlists = dataReader.get_target_playlists()

    recommender = SlimElasticNetRecommender(URM_train)

    evaluator = EvaluatorForSkopt(URM_test, target_playlists, recommender, at=10)

    res = gp_minimize(evaluator.evaluate_slim_skopt, [
        [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550, 600, 700],
        (0.0, 1.0)], n_calls=60, n_random_starts=10, verbose=True)

    print("Final result:")
    print("Best MAP: " + str(res.fun))
    print("Weights: " + str(res.x))
