import numpy as np
from Utilities.ComputeSimilarityPython import ComputeSimilarityPython


class ItemBasedCollaborativeFilteringRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=150, shrink=8, normalize=True):
        similarity_object = ComputeSimilarityPython(self.URM, shrink=shrink, topK=topK, normalize=normalize)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, playlist_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        playlist_profile = self.URM[playlist_id]

        scores = playlist_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(playlist_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, playlist_id):
        playlist_profile = self.URM[playlist_id]

        scores = playlist_profile.dot(self.W_sparse).toarray().ravel()

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        return normalized_scores

    def filter_seen(self, playlist_id, scores):
        start_pos = self.URM.indptr[playlist_id]
        end_pos = self.URM.indptr[playlist_id + 1]

        playlist_profile = self.URM.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores
