import numpy as np
import os

from recommenders.hybrid import HybridRecommender
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython
from misc.data_splitter import train_test_holdout
from misc.submission import create_submission
from misc.evaluator import evaluate_algorithm
from misc.start import create_matrices
from recommenders.matrix_factorization.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython

# creation of the matrices
URM_csr, ICM_csr, targets = create_matrices()

# split the data set
URM_train, URM_test = train_test_holdout(URM_csr, train_perc=0.8)

# initialization and fitting of the hybrid recommender
rec = HybridRecommender(URM_train, ICM_csr, valid=False)
rec.fit_content_based()
rec.fit_item_based()
rec.fit_user_based()
rec.fit_slim_cython(epochs=1, topk=100)

n_users = URM_test.shape[0]
users = np.random.randint(0, n_users, size=10000)

# todo: per testare sulle target tracks al posto di usare 10000 playlist casuali (potrebbe cambiare il risultato ad ogni prova)
# target_tracks = targets["playlist_id"].unique()


#
# reco = MatrixFactorization_FunkSVD_Cython(URM_train)
# reco.fit(epochs=10)
#
#
#
# evaluate_algorithm(URM_test, users, reco)
#
#
#
#rec.fit_MF_cython()

# evaluation of the algorithm
res = []

for num_slim in [1, 2, 3]:
    temp_result = evaluate_algorithm(URM_test, users, rec, num_slim=num_slim)
    res.append(temp_result)

for i in res:
    print(i)

# todo: a che serve?
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 440))
