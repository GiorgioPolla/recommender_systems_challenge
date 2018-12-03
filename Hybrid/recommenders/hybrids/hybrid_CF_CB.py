from recommenders.content_based.content_based import ContentBased
from recommenders.collaborative_filtering.item_based_CF import ItemBasedCF
from recommenders.collaborative_filtering.user_based_CF import UserBasedCF
from misc.others import merge_rankings, merge_prob, filter_seen
import numpy as np


class HybridCFCB(object):
    def __init__(self, urm, icm, alpha=6, beta=3, gamma=2, topk=200, comb='linear'):
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

    def set(self, alpha=6, beta=3, gamma=2, comb='linear'):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.comb = comb

    def get_URM_train(self):
        return self.urm

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        scores_ib = self.ib_rec.recommend(user_id, at=at, only_scores=True)
        scores_cb = self.cb_rec.recommend(user_id, at=at, only_scores=True)
        scores_ub = self.ub_rec.recommend(user_id, at=at, only_scores=True)
        ranking = None

        if self.comb == 'linear':
            scores = np.multiply(scores_ib, self.alpha) + \
                     np.multiply(scores_cb, self.beta) + \
                     np.multiply(scores_ub, self.gamma)
            if only_scores:
                return scores

            if exclude_seen:
                scores = filter_seen(self.urm, user_id, scores)

            ranking = []
            for i in range(np.shape(scores)[0]):
                ranking.append(scores[i].toarray().ravel().argsort()[::-1][:at])

        elif self.comb == 'p_merge':
            ranking = merge_prob(merge_prob(scores_ib, scores_cb, self.beta),
                                 scores_ub, self.gamma, first_ranking=True, exclude_seen=exclude_seen)

        elif self.comb == 'o_merge':
            ranking = merge_rankings(merge_rankings(scores_ib, scores_cb, self.beta),
                                     scores_ub, self.gamma, first_ranking=True, exclude_seen=exclude_seen)

        return ranking
