from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from misc.others import filter_seen
import numpy as np


class UserBasedCF(object):
    def __init__(self, urm):
        self.urm = urm
        self.W = None

    def fit(self, topk=300, shrink=7, normalize=True):
        print('Computing similarity Content Based CF')
        similarity_obj = Compute_Similarity_Cython(self.urm.T, topK=topk, shrink=shrink, normalize=normalize)
        self.W = similarity_obj.compute_similarity()

    def get_URM_train(self):
        return self.urm

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        user_profile = self.W[user_id]
        scores = user_profile.dot(self.urm)

        if only_scores:
            return scores

        if exclude_seen:
            scores = filter_seen(self.urm, user_id, scores)

        ranking = []
        for i in range(np.shape(scores)[0]):
            ranking.append(scores[i].toarray().ravel().argsort()[::-1][:at])

        return ranking
