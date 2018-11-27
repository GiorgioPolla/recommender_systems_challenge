import numpy as np
import scipy .sparse as sps

from sklearn.model_selection import train_test_split


# splitter
def train_test_holdout(URM, train_perc = 0.9):
    num_interactions = URM.nnz
    mat_shape = np.shape(URM)
    URM = URM.tocoo()
    #
    # URM_test, URM_train = train_test_split(URM, train_size=1 - train_perc)

    while True:

        train_mask = np.random.choice([True, False], num_interactions, [train_perc, 1-train_perc])

        URM_train = sps.coo_matrix((URM.data[train_mask], (URM.row[train_mask], URM.col[train_mask])), shape=mat_shape)
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        print("S")
        URM_test = sps.coo_matrix((URM.data[test_mask], (URM.row[test_mask], URM.col[test_mask])), shape=mat_shape)
        URM_test = URM_test.tocsr()
        break

    return URM_train, URM_test
