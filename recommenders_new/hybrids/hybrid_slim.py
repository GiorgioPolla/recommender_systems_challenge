from recommenders.hybrids.hybrid_CF_CB import HybridCFCB
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython
from sklearn.preprocessing import normalize as norm
import numpy as np
from misc.others import merge_rankings, merge_prob, filter_seen, rank


class HybridWithSlim(object):
    def __init__(self, urm, icm, w1=10, w2=800, comb='linear', normalize=False):
        self.urm = urm
        self.hybrid_rec = HybridCFCB(urm, icm)
        self.slim_rec = SLIM_BPR_Cython(norm(urm, norm='l2', axis=1), normalize=normalize)
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def fit(self, epochs=20):
        self.hybrid_rec.fit()
        self.slim_rec.fit(epochs=epochs)

    def set(self, w1=10, w2=5, comb='linear'):
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        scores_hyb = self.hybrid_rec.recommend(user_id, at=at, exclude_seen=exclude_seen, only_scores=True)
        scores_slim = self.slim_rec.recommend(user_id, at=at, exclude_seen=exclude_seen, only_scores=True)

        if self.comb == 'linear':
            scores = self.w1 * scores_hyb + self.w2 * scores_slim
            scores = scores
            if only_scores:
                return scores

            if exclude_seen:
                scores = filter_seen(self.urm, user_id, scores)

            return rank(scores, at=at)

        elif self.comb == 'p_merge':
            ranking = []
            for i in range(np.shape(scores_slim)[0]):
                merge_prob(scores_hyb[i], scores_slim[i], self.w2)
            return ranking

        elif self.comb == 'o_merge':
            return merge_rankings(scores_hyb, scores_slim, self.w2)

        return None
