import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
import os

class EvaluatorForSkopt(object):
    def __init__(self, URM_test, URM_validation, target_playlists, recommender_object, at=10):
        self.target_playlists = target_playlists
        self.recommender_object = recommender_object
        self.at = at
        self.URM_test = sps.csr_matrix(URM_test)
        self.URM_validation = sps.csr_matrix(URM_validation)


    def precision(self, is_relevant, relevant_items):
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        return precision_score


    def recall(self, is_relevant, relevant_items):
        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
        return recall_score


    def MAP(self, is_relevant, relevant_items):
        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        return map_score


    def evaluate_algorithm(self, weights):

        output_root_path = "risultati/"

        # If directory does not exist, create
        if not os.path.exists(output_root_path):
            os.makedirs(output_root_path)

        logFile = open(output_root_path + "risultati.txt", "a")

        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        ib_w, ub_w, cb_w, rp3beta_w, slim_w, mf_w = weights

        print("Evaluating weights: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(ib_w, ub_w, cb_w,rp3beta_w,slim_w,mf_w))

        self.recommender_object.fit(ib_w, ub_w, cb_w, rp3beta_w, slim_w, mf_w)

        # on test
        for user_id in tqdm(self.target_playlists):

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision_test = cumulative_precision / num_eval
        cumulative_recall_test = cumulative_recall / num_eval
        cumulative_MAP_test = cumulative_MAP / num_eval

        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        for user_id in tqdm(self.target_playlists):

            start_pos = self.URM_validation.indptr[user_id]
            end_pos = self.URM_validation.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_validation.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision_validation = cumulative_precision / num_eval
        cumulative_recall_validation = cumulative_recall / num_eval
        cumulative_MAP_validation = cumulative_MAP / num_eval

        precision = (cumulative_precision_test + cumulative_precision_validation) / 2
        recall = (cumulative_recall_test + cumulative_recall_validation) / 2
        MAP = (cumulative_MAP_test + cumulative_MAP_validation) / 2

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            precision, recall, MAP))

        logFile.write("Weights: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}  || MAP: {}\n".format(ib_w, ub_w, cb_w, rp3beta_w, slim_w, mf_w, MAP))
        logFile.flush()

        return -MAP

    def evaluate_algorithm_initial_hybrid(self, weights):

        output_root_path = "risultati/"

        # If directory does not exist, create
        if not os.path.exists(output_root_path):
            os.makedirs(output_root_path)

        logFile = open(output_root_path + "risultati_ibrido.txt", "a")

        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        ib_w, ub_w, cb_w = weights

        self.URM_test = sps.csr_matrix(self.URM_test)

        # n_users = self.URM_test.shape[0]

        print(
            "Evaluating weights: {:.4f},{:.4f},{:.4f}".format(ib_w, ub_w, cb_w))

        self.recommender_object.fit(ib_w, ub_w, cb_w)

        for user_id in tqdm(self.target_playlists):

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

        logFile.write(
            "Pesi: {:.4f}, {:.4f}, {:.4f}  || MAP: {}\n".format(ib_w, ub_w, cb_w, cumulative_MAP))
        logFile.flush()

        return -cumulative_MAP

    def evaluate_hybrid_cb_gb(self, weights):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        cb_w, rp3beta_w = weights

        self.URM_test = sps.csr_matrix(self.URM_test)

        n_users = self.URM_test.shape[0]

        for user_id in self.target_playlists:

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, cb_w, rp3beta_w, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

        result_dict = {
            "precision": cumulative_precision,
            "recall": cumulative_recall,
            "MAP": cumulative_MAP,
        }

        return -cumulative_MAP

    def evaluate_hybrid_cb_gb_ub(self, weights):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        cb_w, ub_w, rp3beta_w = weights

        self.URM_test = sps.csr_matrix(self.URM_test)

        n_users = self.URM_test.shape[0]

        for user_id in self.target_playlists:

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, cb_w, ub_w, rp3beta_w, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

        result_dict = {
            "precision": cumulative_precision,
            "recall": cumulative_recall,
            "MAP": cumulative_MAP,
        }

        return -cumulative_MAP

    def evaluate_slim_skopt(self, parameters):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        l1_ratio, topk = parameters

        self.URM_test = sps.csr_matrix(self.URM_test)

        n_users = self.URM_test.shape[0]

        self.recommender_object.fit(topk, True, l1_ratio)

        for user_id in self.target_playlists:

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))
        print("Weights: " + str(l1_ratio) + " " + str(topk) )

        result_dict = {
            "precision": cumulative_precision,
            "recall": cumulative_recall,
            "MAP": cumulative_MAP,
        }

        return -cumulative_MAP

    def evaluate_hybrid(self, parameters):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        ib_w, ub_w, cb_w, rp3beta_w, slim_w, l1_ratio, topk = parameters

        self.URM_test = sps.csr_matrix(self.URM_test)

        n_users = self.URM_test.shape[0]

        self.recommender_object.fit(ib_w, ub_w, cb_w, rp3beta_w, slim_w, topk, l1_ratio)

        for user_id in self.target_playlists:

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, at=self.at, exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))
        print("Weights: " + str(ib_w) + " " + str(ub_w) + " " + str(cb_w) + " " + str(rp3beta_w) + " " + str(slim_w) + " " + str(l1_ratio) + " " + str(topk) )

        result_dict = {
            "precision": cumulative_precision,
            "recall": cumulative_recall,
            "MAP": cumulative_MAP,
        }

        return -cumulative_MAP

    def evaluate_algorithm_last(self, weights):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        num_eval = 0

        ib_w, cb_w, ub_w, rp3beta_w, slim_w, topK_ib, topK_cb, topK_ub, topK_rp3beta, topK_slim, shrink_ib, shrink_cb, shrink_ub, alpha, beta, l1_ratio = weights

        self.URM_test = sps.csr_matrix(self.URM_test)

        n_users = self.URM_test.shape[0]

        self.recommender_object.fit(topK_ib, topK_cb, topK_ub, topK_rp3beta, topK_slim, shrink_ib, shrink_cb, shrink_ub, alpha, beta, l1_ratio)

        for user_id in self.target_playlists:

            start_pos = self.URM_test.indptr[user_id]
            end_pos = self.URM_test.indptr[user_id + 1]

            if end_pos - start_pos > 0:
                relevant_items = self.URM_test.indices[start_pos:end_pos]

                recommended_items = self.recommender_object.recommend(user_id, ib_w, cb_w, ub_w, rp3beta_w, slim_w, at=self.at,
                                                                      exclude_seen=True)
                num_eval += 1

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                cumulative_precision += self.precision(is_relevant, relevant_items)
                cumulative_recall += self.recall(is_relevant, relevant_items)
                cumulative_MAP += self.MAP(is_relevant, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))

        print("Weights: " + str(ib_w) + " " + str(cb_w) + " " + str(ub_w) + " " + str(rp3beta_w) + " " + str(slim_w) + "\n" +
              "TopK: " + str(topK_ib) + " " + str(topK_cb) + " " + str(topK_ub) + " " + str(topK_rp3beta) + " " + str(topK_slim) + "\n" +
              "Shrink: " + str(shrink_ib) + " " + str(shrink_cb) + " " + str(shrink_ub) + "\n"
              "Alpha: " + str(alpha) + " Beta: " + str(beta) + " l1_ratio: " + str(l1_ratio))

        result_dict = {
            "precision": cumulative_precision,
            "recall": cumulative_recall,
            "MAP": cumulative_MAP,
        }

        return -cumulative_MAP