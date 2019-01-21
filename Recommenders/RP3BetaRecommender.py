import numpy as np
import scipy.sparse as sps
import time
import sys
from sklearn.preprocessing import normalize
from Utilities.Support import check_matrix


class RP3betaRecommender(object):

    def __init__(self, URM):
        self.URM = check_matrix(URM, format='csr', dtype=np.float32)

    def fit(self, alpha=0.5766, beta=0.2379, min_rating=0, topK=80, implicit=True, normalize_similarity=True):

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM.data[self.URM.data < self.min_rating] = 0
            self.URM.eliminate_zeros()
            if self.implicit:
                self.URM.data = np.ones(self.URM.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W = normalize(self.W, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = self.similarityMatrixTopK(self.W, forceSparseOutput=True, k=self.topK)
            self.sparse_weights = True

    def recommend(self, playlist_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        playlist_profile = self.URM[playlist_id]

        scores = playlist_profile.dot(self.W_sparse).toarray().ravel()

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        if exclude_seen:
            scores = self.filter_seen(playlist_id, normalized_scores)

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

    def similarityMatrixTopK(self, item_weights, forceSparseOutput=True, k=150, verbose=False, inplace=True):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param forceSparseOutput:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if forceSparseOutput:
                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):
                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx + 1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                non_zero_data = column_data != 0

                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

            # During testing CSR is faster
            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse
