import os

from recommenders.collaborative_filtering.item_based_CF import ItemBasedCF
from misc.data_splitter import train_test_holdout
from misc.start import create_matrices

from ParameterTuning.AbstractClassSearch import EvaluatorWrapper, DictionaryKeys
from Base.Evaluation.Evaluator import SequentialEvaluator
from ParameterTuning.BayesianSearch import BayesianSearch

urm_csr, icm_csr, targets = create_matrices()
urm_train, urm_test, _, ignored = train_test_holdout(urm_csr, train_perc=0.8)
urm_train, urm_validation, _, ign = train_test_holdout(urm_train, train_perc=0.9)

ev_valid = SequentialEvaluator(urm_validation, cutoff_list=[10], ignore_users=ign)
ev_test = SequentialEvaluator(urm_test, cutoff_list=[10], ignore_users=ignored)

ev_valid = EvaluatorWrapper(ev_valid)
ev_test = EvaluatorWrapper(ev_test)

rec = ItemBasedCF
param_search = BayesianSearch(rec, evaluator_validation=ev_valid, evaluator_test=ev_test)

p_range = {}
p_range['topk'] = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800]
p_range['shrink'] = [0, 5, 10, 25, 50, 100, 150, 200, 300, 500]
p_range['similarity'] = ['cosine']
p_range['normalize'] = [True]

dictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [urm_train],
              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: p_range}

output_root_path = "result_experiments/"

if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

output_root_path += 'Item Based'

n = 35
metric = 'MAP'
best = param_search.search(dictionary, n_cases=n, output_root_path=output_root_path,
                           metric=metric)
