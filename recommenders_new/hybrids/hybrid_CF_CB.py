from recommenders.content_based.content_based import ContentBased
from recommenders.collaborative_filtering.item_based_CF import ItemBasedCF
from recommenders.collaborative_filtering.user_based_CF import UserBasedCF
from misc.others import merge_rankings, merge_prob, filter_seen, rank
import numpy as np
import scipy.sparse as sps

#
#   alpha = 10
#   beta = 5
#   gamma = 3
#


class HybridCFCB(object):
    def __init__(self, urm, icm, alpha=10, beta=5, gamma=3, comb='linear'):
        self.urm = urm
        self.ib_rec = ItemBasedCF(urm)
        self.cb_rec = ContentBased(urm, icm)
        self.ub_rec = UserBasedCF(urm)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.comb = comb

    def fit(self):
        self.ib_rec.fit()
        self.ub_rec.fit()
        self.cb_rec.fit()

    def set(self, alpha=6, beta=4, gamma=1, comb='linear'):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.comb = comb

    def linear_comb(self, scores_ib, scores_cb, scores_ub):
        return scores_ib * self.alpha + scores_cb * self.beta + scores_ub * self.gamma

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        scores_ib = self.ib_rec.recommend(user_id, at=at, only_scores=True)
        scores_cb = self.cb_rec.recommend(user_id, at=at, only_scores=True)
        scores_ub = self.ub_rec.recommend(user_id, at=at, only_scores=True)

        if self.comb == 'linear':
            scores = self.linear_comb(scores_ib, scores_cb, scores_ub)

            if only_scores:
                return scores

            if exclude_seen:
                scores = filter_seen(self.urm, user_id, scores)

            return rank(scores, at=at)

        # elif self.comb == 'p_merge':
        #     return merge_prob(merge_prob(scores_ib, scores_cb, self.beta),
        #                       scores_ub, self.gamma, first_ranking=True, exclude_seen=exclude_seen)
        #
        # elif self.comb == 'o_merge':
        #     return merge_rankings(merge_rankings(scores_ib, scores_cb, self.beta),
        #                           scores_ub, self.gamma, first_ranking=True, exclude_seen=exclude_seen)

        return None
