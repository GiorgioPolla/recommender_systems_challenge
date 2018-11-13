import numpy as np
import pandas as pd
import scipy.sparse as sps
import os
# imports the class to evaluate
from top_pop import Recommender


class Evaluator(object):

    def read_file(self):
        current_path = os.path.dirname(os.path.abspath('__file__'))
        train_file = current_path + '/data/train.csv'

        train_data = pd.read_csv(train_file)
        train_data.columns = ['playlist_id', 'track_id']

        return train_data

    def precision(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

        return precision_score

    def recall(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

        return recall_score


    def MAP(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def evaluate_algorithm(self, URM_test, recommender_object, at):

        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        URM_test = sps.csr_matrix(URM_test)

        n_users = URM_test.shape[0]


        for user_id in range(n_users):

            if user_id % 10000 == 0:
                print("Evaluated user {} of {}".format(user_id, n_users))

            start_pos = URM_test.indptr[user_id]
            end_pos = URM_test.indptr[user_id+1]

            if end_pos-start_pos>0:

                relevant_items = URM_test.indices[start_pos:end_pos]

                recommended_items = recommender_object.recommend(user_id, at)
                num_eval+=1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)


        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

    def create_URMs(self, train_data):
        train_test_split = 0.865

        userList = train_data['playlist_id'].values
        itemList = train_data['track_id'].values
        ratingList = np.full(len(userList), 1)

        train_mask = np.random.choice([1, 0], len(userList), p=[train_test_split, 1 - train_test_split])

        URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
        URM_test = URM_test.tocsr()

        return URM_train, URM_test

    def evaluate(self):

        rec = Recommender()

        train_data = self.read_file()

        URM_train, URM_test = self.create_URMs(train_data)

        rec.fit(URM_train)

        self.evaluate_algorithm(URM_test, rec, 10)

evaluator = Evaluator()

evaluator.evaluate()