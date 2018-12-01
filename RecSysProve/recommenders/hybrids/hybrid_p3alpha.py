from recommenders.hybrids.hybrid_CF_CB import HybridCFCB
from recommenders.p3alpha.P3alphaRecommender import P3alphaRecommender
from misc.others import filter_seen


class HybridWithP3alpha(object):
    def __init__(self, urm, icm, w1=1, w2=60, comb='linear'):
        self.urm = urm
        self.hybrid_rec = HybridCFCB(urm, icm)
        self.p3alpha_rec = P3alphaRecommender(urm)
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def fit(self):
        self.hybrid_rec.fit()
        self.p3alpha_rec.fit()

    def set(self, w1=1, w2=60, comb='linear'):
        self.w1 = w1
        self.w2 = w2
        self.comb = comb

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        scores_hyb = self.hybrid_rec.recommend(user_id, at=at, only_scores=True)
        scores_slim = self.p3alpha_rec.recommend(user_id, at=at, only_scores=True)
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
