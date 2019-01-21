from Dataset.DatasetReader import DatasetReader
from Recommenders.HybridRecommender import HybridRecommender
from Utilities.Support import create_submission

if __name__ == '__main__':
    dataReader = DatasetReader()

    URM = dataReader.get_URM()
    ICM = dataReader.get_ICM()
    target_playlists = dataReader.get_target_playlists()

    print("Making recommendations...")

    recommender = HybridRecommender(URM, ICM)

    recommender.fit(0.0922, 0.1996, 0.2374, 0.4319, 0.35, 0.5619)

    create_submission(recommender, target_playlists, 10)
