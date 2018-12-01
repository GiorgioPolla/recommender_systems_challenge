import os

from recommenders.hybrids.old_hybrid import HybridRecommender
from recommenders.hybrids.hybrid_CF_CB import HybridCFCB
from misc.data_splitter import train_test_holdout
from misc.evaluator import evaluate_algorithm
from misc.start import create_matrices


URM_csr, ICM_csr, targets = create_matrices()

URM_train, URM_test, users = train_test_holdout(URM_csr, train_perc=0.8)

rec = HybridCFCB(URM_train, ICM_csr)

rec.fit()

res = []

for i in [0, 0.17]:
    temp_result = evaluate_algorithm(URM_test, users, rec)

    res.append(((i), temp_result))

for i in res:
    print(i)

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 440))
