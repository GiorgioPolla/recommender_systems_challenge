import numpy as np
import scipy .sparse as sps


# splitter
def train_test_holdout(URM, train_perc=0.8):

    num_interactions = URM.nnz
    mat_shape = np.shape(URM)
    URM = URM.tocoo()

    train_mask = np.random.choice([True, False], num_interactions, [train_perc, 1-train_perc])

    URM_train = sps.coo_matrix((URM.data[train_mask], (URM.row[train_mask], URM.col[train_mask])), shape=mat_shape)
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM.data[test_mask], (URM.row[test_mask], URM.col[test_mask])), shape=mat_shape)
    URM_test = URM_test.tocsr()

    return URM_train, URM_test
