import scipy.sparse as sps
import numpy as np
from tqdm import tqdm

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)

# split dataset
def split_train_validation_test(URM_all, split_train_test_validation_quota):

    URM_all = URM_all.tocoo()
    URM_shape = URM_all.shape

    numInteractions= len(URM_all.data)

    split = np.random.choice([1, 2, 3], numInteractions, p=split_train_test_validation_quota)


    trainMask = split == 1
    URM_train = sps.coo_matrix((URM_all.data[trainMask], (URM_all.row[trainMask], URM_all.col[trainMask])), shape=URM_shape)
    URM_train = URM_train.tocsr()

    testMask = split == 2

    URM_test = sps.coo_matrix((URM_all.data[testMask], (URM_all.row[testMask], URM_all.col[testMask])), shape=URM_shape)
    URM_test = URM_test.tocsr()

    validationMask = split == 3

    URM_validation = sps.coo_matrix((URM_all.data[validationMask], (URM_all.row[validationMask], URM_all.col[validationMask])), shape=URM_shape)
    URM_validation = URM_validation.tocsr()


    return URM_train, URM_validation, URM_test


# Evaluator
def precision(is_relevant, relevant_items):
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score


def recall(is_relevant, relevant_items):
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score


def MAP(is_relevant, relevant_items):
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score


def evaluate_algorithm(URM_test, target_playlists, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    for user_id in tqdm(target_playlists):

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict

# make recommendations
def initialize_output_file():
    file = open("submission.csv", 'a')
    file.write("playlist_id,track_ids" + '\n')
    return file


def print_to_file(playlist, tracks, file):
    file.write(str(playlist) + ',')
    index = 0
    while index < 9:
        file.write(str(tracks[index]) + ' ')
        index += 1
    file.write(str(tracks[index]) + '\n')


def create_submission(recommender_object, targets,  at=10):
    file = initialize_output_file()

    for playlist in tqdm(targets):
        recommended_items = recommender_object.recommend(playlist, at)
        print_to_file(playlist, recommended_items, file)

    file.close()
