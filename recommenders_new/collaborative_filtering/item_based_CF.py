from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from misc.others import filter_seen, rank


class ItemBasedCF(object):
    def __init__(self, urm):
        self.urm = urm
        self.W = None

# topk = 600
# shrink = 30
# similarity = dice

    def fit(self, topk=600, shrink=30, normalize=True, similarity='dice'):
        print('Computing similarity Item Based CF')

        similarity_obj = Compute_Similarity_Cython(self.urm, topK=topk, shrink=shrink,
                                                   normalize=normalize, similarity=similarity)
        self.W = similarity_obj.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=True, only_scores=False):
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.W)

        if only_scores:
            return scores

        if exclude_seen:
            scores = filter_seen(self.urm, user_id, scores)

        return rank(scores, at=at)
