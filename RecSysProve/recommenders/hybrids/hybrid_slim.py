from recommenders.hybrids.hybrid_CF_CB import HybridCFCB
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython
import numpy as np
from misc.others import merge_rankings, merge_prob, filter_seen


class HybridWithSlim(object):
    def __init__(self, urm, icm, w1=8, w2=10, comb='linear'):
        self.urm = urm
        self.hybrid_rec = HybridCFCB(urm, icm)
        self.slim_rec = SLIM_BPR_Cython(urm)
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def fit(self):
        self.hybrid_rec.fit()
        self.slim_rec.fit(epochs=1)

    def set(self, w1=8, w2=10, comb='linear'):
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        scores_hyb = self.hybrid_rec.recommend(user_id, at=at, exclude_seen=exclude_seen, only_scores=True)
        scores_slim = self.slim_rec.recommend(user_id, at=at, exclude_seen=exclude_seen, only_scores=True)
        ranking = None

        if self.comb == 'linear':
            scores = self.w1 * scores_hyb + self.w2 * scores_slim
            if only_scores:
                return scores

            if exclude_seen:
                scores = filter_seen(self.urm, user_id, scores)

            ranking = scores.argsort()[::-1][:at]

        # elif self.comb == 'p_merge':
        #     ranking = merge_prob(scores_hyb, scores_slim, self.w2),
        #
        # elif self.comb == 'o_merge':
        #     ranking = merge_rankings(scores_hyb, scores_slim, self.w2)

        return ranking
