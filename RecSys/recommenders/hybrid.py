import numpy as np

from misc.similarity import Compute_Similarity_Python
from recommenders.slim.standard_slim import StandardSlim
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython
from recommenders.matrix_factorization.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython as MF
from misc.data_splitter import train_test_holdout

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

    def __init__(self, URM, ICM, valid=False):
        self.URM = URM
        self.ICM = ICM
        #self.MF_cython = MF(URM, recompile_cython=True)
        self.standard_slim = StandardSlim(URM)
        self.valid = valid
        if valid:
            URM_train_slim, URM_valid = train_test_holdout(URM, train_perc=0.8)
            self.slim_cython = SLIM_BPR_Cython(URM_train_slim, URM_validation=URM_valid)
        else:
            self.slim_cython = SLIM_BPR_Cython(URM)

    def fit_MF_cython(self):
        self.MF_cython.fit()

    def fit_slim_cython(self, epochs=300, topk=200, sgd_mode='adagrad', beta1=0.9, beta2=0.999):
        self.W_slim_cython = self.slim_cython.fit(stop_on_validation=self.valid, epochs=epochs, topK=topk, sgd_mode=sgd_mode, beta_1=beta1, beta_2=beta2)

    def fit_slim(self, learning_rate=0.01, epochs=35):
        self.W_slim = self.standard_slim.fit(learning_rate, epochs)

    def fit_content_based(self, topK=100, shrink=5, normalize=True, similarity="cosine"):

        similarity_object_content_based = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
                                                                    topK=topK, normalize=normalize,
                                                                    similarity=similarity)

        self.W_sparse_content_based = similarity_object_content_based.compute_similarity()

    def fit_item_based(self, topK=100, shrink=5, normalize=True, similarity="cosine"):

        similarity_object_item_cf = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                              topK=topK, normalize=normalize,
                                                              similarity=similarity)

        self.W_sparse_item_cf = similarity_object_item_cf.compute_similarity()

    def fit_user_based(self, topK=100, shrink=5, normalize=True, similarity="cosine"):

        similarity_object_user_based = Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                                 topK=topK, normalize=normalize,
                                                                 similarity=similarity)

        self.W_sparse_user_cf = similarity_object_user_based.compute_similarity()

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommend(self, user_id, alfa=6, beta=3, gamma=30, at=10, num_slim=0, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        user_profile_ub = self.W_sparse_user_cf[user_id]
        # #
        scores_content_based = user_profile.dot(self.W_sparse_content_based).toarray().ravel()
        scores_item_cf = user_profile.dot(self.W_sparse_item_cf).toarray().ravel()
        scores_user_cf = user_profile_ub.dot(self.URM).toarray().ravel()
        scores_slim_cython = user_profile.dot(self.W_slim_cython).toarray().ravel()
        #scores = user_profile.dot(self.W_slim_cython).toarray().ravel()

        # #scores =(1 - gamma)*((1 - beta)*((1 - alfa)*scores_content_based + alfa*scores_item_cf) + beta*scores_user_cf) +gamma * scores_slim
        # # #
        scores_not_slim = beta * scores_content_based + alfa * scores_item_cf + scores_user_cf
        # # # #

        if exclude_seen:
            scores_slim_cython = self.filter_seen(user_id, scores_slim_cython)
            scores_not_slim = self.filter_seen(user_id, scores_not_slim)
            #scores = self.filter_seen(user_id, scores)

        # rank items
        # ranking = scores.argsort()[::-1]

        ranking_not_slim = scores_not_slim.argsort()[::-1]

        ranking_slim_cython = scores_slim_cython.argsort()[::-1][:at * 20]

        ranking = ranking_not_slim[:at]

        j = 0
        i = 0
        while i < num_slim:
            if ranking_slim_cython[i+j] in ranking[:10 - num_slim + i]:
                j += 1
            else:
                ranking[10 - num_slim + i] = ranking_slim_cython[i + j]
                i += 1

        return ranking[:at]
