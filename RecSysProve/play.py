from recommenders.hybrids.hybrid_p3alpha import HybridWithP3alpha
from recommenders.slim.cython.slim_BPR_cython import SLIM_BPR_Cython
from recommenders.hybrids.hybrid_CF_CB import HybridCFCB
from Base.Evaluation.Evaluator import SequentialEvaluator
from recommenders.collaborative_filtering.user_based_CF import UserBasedCF
from recommenders.collaborative_filtering.item_based_CF import ItemBasedCF
from recommenders.content_based.content_based import ContentBased
from misc.data_splitter import train_test_holdout
from misc.evaluator import evaluate_algorithm
from misc.start import create_matrices
from misc.submission import create_submission


urm_csr, icm_csr, targets = create_matrices()
urm_train, urm_test, users = train_test_holdout(urm_csr, train_perc=0.8)
rec = HybridCFCB(urm_train, icm_csr)
ev = SequentialEvaluator(urm_test, [10], random_users=10000)

rec.fit()

res = []
for a, b, g in [(1, 1, 1), (10, 1, 1), (6, 3, 2)]:
    rec.set(alpha=a, beta=b, gamma=g)
    #temp_result = evaluate_algorithm(urm_test, users, rec)
    dict, _ = ev.evaluateRecommender(rec)
    print(dict)
    #res.append((i, temp_result))

for i in res:
    print(i)
