import numpy as np

from misc.similarity import Compute_Similarity_Python
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython

'''
cb_shrink = 5  # 5
ib_shrink = 15  # 20
ub_shrink = 7  # 7

cb_topk = 300  # 300
ib_topk = 200  # 200
ub_topk = 300  # 300

epochs = 40  # 40

alfa = 0.6 
beta = 0.1
gamma = 0.1  # 0.1
'''


class HybridRecommender(object):

    def __init__(self, URM, ICM, CF_CB=False, slim_cyt=False, MF_cyt=False):
        self.URM = URM
        self.ICM = ICM

        self.CF_CB = CF_CB
        self.slim_cyt = slim_cyt
        self.MF_cyt = MF_cyt

        if slim_cyt:
            self.slim = SLIM_BPR_Cython(URM)

    #
    def fit_MF(self, epochs=300, num_factors=10):
        return

    #
    def fit_slim(self, epochs=300, topk=100):
        if self.slim_cyt:
            self.W_slim = self.slim.fit(epochs=epochs, topK=topk)

    #
    def fit_content_based(self, topK=200, shrink=5, normalize=True, similarity="cosine"):
        if self.CF_CB:
            similarity_object_content_based = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                                        topK=topK, normalize=normalize,
                                                                        similarity=similarity)
            print("Computing similarity for Content Based")
            self.W_sparse_content_based = similarity_object_content_based.compute_similarity()

    #
    def fit_item_based(self, topK=200, shrink=10, normalize=True, similarity="cosine"):
        if self.CF_CB:
            similarity_object_item_cf = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                                  topK=topK, normalize=normalize,
                                                                  similarity=similarity)
            print("Computing similarity for Item Based")
            self.W_sparse_item_cf = similarity_object_item_cf.compute_similarity()

    #
    def fit_user_based(self, topK=200, shrink=7, normalize=True, similarity="cosine"):
        if self.CF_CB:
            similarity_object_user_based = Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                                     topK=topK, normalize=normalize,
                                                                     similarity=similarity)
            print("Computing similarity for User Based")
            self.W_sparse_user_cf = similarity_object_user_based.compute_similarity()

    #
    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    #
    def merge_rankings(self, ranking_one, ranking_two, num_two, at=10):
        j = 0
        i = 0
        while i < num_two:
            if i + j == len(ranking_two):
                break
            if at - num_two + i == len(ranking_one):
                k = 0
                while len(ranking_one) < at and i + j + k < len(ranking_two):
                    ranking_one = np.append(ranking_one, ranking_two[i + j + k])
                    k += 1
                break
            if ranking_two[i + j] in ranking_one[:at - num_two]:
                j += 1
            else:
                ranking_one = np.insert(ranking_one, [at - num_two + i], ranking_two[i + j])
                i += 1
        return ranking_one[:at]

    #
    def merge_probab(self, ranking_one, ranking_two, prob_two, at=10):
        ranking = []
        c_one = 0
        c_two = 0

        while len(ranking) < at and (c_one < len(ranking_one) or c_two < len(ranking_two)):
            n_rand = np.random.random_sample()

            if c_one == len(ranking_one) or n_rand < prob_two:
                while c_two < len(ranking_two):
                    if ranking_two[c_two] in ranking:
                        c_two += 1
                    else:
                        ranking.append(ranking_two[c_two])
                        c_two += 1
                        break

            else:
                while c_one < len(ranking_one):
                    if ranking_one[c_one] in ranking:
                        c_one += 1
                    else:
                        ranking.append(ranking_one[c_one])
                        c_one += 1
                        break

        return ranking
    #
    def recommend(self, user_id, alfa=6, beta=3, gamma=1.6, at=10, num_slim=0.17, exclude_seen=True):
        user_profile = self.URM[user_id]

        # Content Based and the two Collaborative Filtering
        if self.CF_CB:
            user_profile_ub = self.W_sparse_user_cf[user_id]

            scores_content_based = user_profile.dot(self.W_sparse_content_based).toarray().ravel()
            scores_item_cf = user_profile.dot(self.W_sparse_item_cf).toarray().ravel()
            scores_user_cf = user_profile_ub.dot(self.URM).toarray().ravel()
            scores_CB_CF = beta * scores_content_based + alfa * scores_item_cf + gamma * scores_user_cf

            if exclude_seen:
                scores_CB_CF = self.filter_seen(user_id, scores_CB_CF)

            ranking_CB_CF = scores_CB_CF.argsort()[::-1]

            if not self.slim_cyt and not self.MF_cyt:
                return ranking_CB_CF[:at]

        # Matrix factorization
        # if self.MF_cyt:
        #     scores_MF = self.MF.compute_score_MF(user_id)
        #
        #     if exclude_seen:
        #         scores_MF = self.filter_seen(user_id, scores_MF)
        #
        #     ranking_MF = scores_MF.argsort()[::-1]
        #
        #     if not self.slim_cyt and not self.CF_CB:
        #         return ranking_MF[:at]

        # SLIM
        if self.slim_cyt:
            scores_slim = user_profile.dot(self.W_slim).toarray().ravel()

            if exclude_seen:
                scores_slim = self.filter_seen(user_id, scores_slim)

            ranking_slim = scores_slim.argsort()[::-1]

            if not self.CF_CB and not self.MF_cyt:
                return ranking_slim[:at]

        if self.slim_cyt and not self.MF_cyt and self.CF_CB:
            if(isinstance(num_slim, int)):
                return self.merge_rankings(ranking_CB_CF, ranking_slim, num_slim)
            else:
                return self.merge_probab(ranking_CB_CF, ranking_slim, num_slim)

        # if exclude_seen:
            # scores = self.filter_seen(user_id, scores)

        # rank items
        # ranking = scores.argsort()[::-1]

        return None
