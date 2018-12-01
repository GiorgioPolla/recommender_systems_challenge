import numpy as np


def filter_seen(urm, user_id_array, scores):
    for el in range(len(user_id_array)):
        user_id = user_id_array[el]
        start_pos = urm.indptr[user_id]
        end_pos = urm.indptr[user_id + 1]

        user_profile = urm.indices[start_pos:end_pos]

        scores[el][0, user_profile] = -np.inf

    return scores


#
def merge_rankings(scores_one, scores_two, num_two, at=10, first_ranking=False, second_ranking=False, exclude_seen=False):
    if not first_ranking:
        ranking_one = scores_one.argsort()[::-1]
    else:
        ranking_one = scores_one
    if not second_ranking:
        ranking_two = scores_two.argsort()[::-1]
    else:
        ranking_two = scores_two
    i = 0
    j = 0

    while i < num_two:
        if i + j == len(ranking_two):
            break
        if at - num_two + i == len(ranking_one):
            k = 0
            while len(ranking_one) < at and i + j + k < len(ranking_two):
                ranking_one = np.append(ranking_one, ranking_two[i + j + k])
                k += 1
            break
        if ranking_two[i + j] in ranking_one[:at - num_two]:
            j += 1
        else:
            ranking_one = np.insert(ranking_one, [at - num_two + i], ranking_two[i + j])
            i += 1
    return ranking_one[:at]


#
def merge_prob(scores_one, scores_two, prob_two, at=10, first_ranking=False, second_ranking=False, exclude_seend=False):
    if not first_ranking:
        ranking_one = scores_one.argsort()[::-1]
    else:
        ranking_one = scores_one
    if not second_ranking:
        ranking_two = scores_two.argsort()[::-1]
    else:
        ranking_two = scores_two
    ranking = []
    c_one = 0
    c_two = 0

    while len(ranking) < at and (c_one < len(ranking_one) or c_two < len(ranking_two)):
        n_rand = np.random.random_sample()

        if c_one == len(ranking_one) or n_rand < prob_two:
            while c_two < len(ranking_two):
                if ranking_two[c_two] in ranking:
                    c_two += 1
                else:
                    ranking.append(ranking_two[c_two])
                    c_two += 1
                    break

        else:
            while c_one < len(ranking_one):
                if ranking_one[c_one] in ranking:
                    c_one += 1
                else:
                    ranking.append(ranking_one[c_one])
                    c_one += 1
                    break

    return ranking
