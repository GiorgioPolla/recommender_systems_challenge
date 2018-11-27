import numpy as np
import os

from recommenders.hybrid import HybridRecommender
from misc.data_splitter import train_test_holdout
from misc.submission import create_submission
from misc.evaluator import evaluate_algorithm
from misc.start import create_matrices


URM_csr, ICM_csr, targets = create_matrices()

URM_train, URM_test = train_test_holdout(URM_csr, train_perc=0.8)

rec = HybridRecommender(URM_train, ICM_csr, CF_CB=True, slim_cyt=True, MF_cyt=False)

rec.fit_content_based()
rec.fit_user_based()
rec.fit_item_based()
rec.fit_slim(epochs=30)

n_users = URM_test.shape[0]
users = np.random.randint(0, n_users, size=10000)


res = []

for i in []:
    temp_result = evaluate_algorithm(URM_test, users, rec, num_slim=i)

    res.append(((i), temp_result))

for i in res:
    print(i)

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 440))
