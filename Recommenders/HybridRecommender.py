import numpy as np

from Recommenders.RP3BetaRecommender import RP3betaRecommender
from Recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.MatrixFactorizationRecommender import MatrixFactorizationRecommender
from Recommenders.ItemBasedCollaborativeFilteringRecommender import ItemBasedCollaborativeFilteringRecommender
from Recommenders.UserBasedCollaborativeFilteringRecommender import UserBasedCollaborativeFilteringRecommender
from Recommenders.ContentBasedRecommender import ContentBasedRecommender


class HybridRecommender(object):
    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

        self.rp3beta = RP3betaRecommender(URM)
        self.slim = SLIM_BPR_Cython(URM)
        self.ib = ItemBasedCollaborativeFilteringRecommender(URM)
        self.ub = UserBasedCollaborativeFilteringRecommender(URM)
        self.cb = ContentBasedRecommender(URM, ICM)
        self.mf = MatrixFactorizationRecommender(URM)

        self.ub.fit()
        self.ib.fit()
        self.cb.fit()
        self.rp3beta.fit()
        self.slim.fit()
        self.mf.fit()

    def fit(self, ib_w=1, ub_w=1, cb_w=1, rp3beta_w=1, slim_w=1, mf_w=1):
        self.rp3beta_w = rp3beta_w
        self.slim_w = slim_w
        self.mf_w = mf_w
        self.ib_w = ib_w
        self.ub_w = ub_w
        self.cb_w = cb_w

    def recommend(self, playlist_id, at=None, exclude_seen=True):
        self.rp3beta_scores = self.rp3beta.get_scores(playlist_id)
        self.slim_scores = self.slim.get_scores(playlist_id)
        self.ib_scores = self.ib.get_scores(playlist_id)
        self.ub_scores = self.ub.get_scores(playlist_id)
        self.cb_scores = self.cb.get_scores(playlist_id)
        self.mf_scores = self.mf.get_scores(playlist_id)

        scores = self.ib_scores * self.ib_w + self.ub_scores * self.ub_w + self.cb_scores * self.cb_w + self.rp3beta_scores * self.rp3beta_w + self.slim_scores * self.slim_w + self.mf_scores * self.mf_w

        if exclude_seen:
            scores = self.filter_seen(playlist_id, scores)

        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, playlist_id, scores):
        start_pos = self.URM.indptr[playlist_id]
        end_pos = self.URM.indptr[playlist_id + 1]

        playlist_profile = self.URM.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores
