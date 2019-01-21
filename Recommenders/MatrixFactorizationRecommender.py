import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


class MatrixFactorizationRecommender(object):

    # num_factors=290, reg=0.035, iters=20 about the same result
    def __init__(self, URM_train,
                 num_factors=350,
                 reg=0.015,
                 iters=20,
                 scaling='log',
                 alpha=40,
                 epsilon=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: scaling factor to compute confidence scores
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

        self.dataset = URM_train
        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.scaling = scaling
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def _linear_scaling(self, R):
        C = R.copy().tocsr()
        C.data *= self.alpha
        np.add(C.data, 1.0, out=C.data, casting="unsafe")
        return C

    def _log_scaling(self, R):
        C = R.copy().tocsr()
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C

    def fit(self):

        test = False
        modelName = "1"

        # compute the confidence matrix

        print("DatasetReader: loading mf data...")

        dataSubfolder = "./Dataset/"

        if (test):
            try:
                self.X = np.load(dataSubfolder + "X_" + modelName + ".dat")
                self.Y = np.load(dataSubfolder + "Y_" + modelName + ".dat")

            except FileNotFoundError:

                print("DatasetReader: building new mf matrices")

                if self.scaling == 'linear':
                    C = self._linear_scaling(self.dataset)
                else:
                    C = self._log_scaling(self.dataset)

                Ct = C.T.tocsr()
                M, N = self.dataset.shape

                # set the seed
                np.random.seed(self.rnd_seed)

                # initialize the latent factors
                self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
                self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

                for it in tqdm(range(self.iters)):
                    self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
                    self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
                    logger.debug('Finished iter {}'.format(it + 1))

                self.X.dump(dataSubfolder + "X_" + modelName + ".dat")
                self.Y.dump(dataSubfolder + "Y_" + modelName + ".dat")
        else:
            try:
                self.X = np.load(dataSubfolder + "AllX.dat")
                self.Y = np.load(dataSubfolder + "AllY.dat")

            except FileNotFoundError:

                print("DatasetReader: building new mf matrices")

                if self.scaling == 'linear':
                    C = self._linear_scaling(self.dataset)
                else:
                    C = self._log_scaling(self.dataset)

                Ct = C.T.tocsr()
                M, N = self.dataset.shape

                # set the seed
                np.random.seed(self.rnd_seed)

                # initialize the latent factors
                self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
                self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

                for it in tqdm(range(self.iters)):
                    self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
                    self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
                    logger.debug('Finished iter {}'.format(it + 1))

                self.X.dump(dataSubfolder + "AllX.dat")
                self.Y.dump(dataSubfolder + "AllY.dat")
        print("DatasetReader: loading complete")

    def recommend(self, playlist_id, at=None, remove_seen_flag=True):
        scores = np.dot(self.X[playlist_id], self.Y.T)

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        if remove_seen_flag:
            scores = self.filter_seen(playlist_id, normalized_scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, playlist_id):
        scores = np.dot(self.X[playlist_id], self.Y.T)

        maximum = np.amax(scores)

        normalized_scores = np.true_divide(scores, maximum)

        return normalized_scores

    def _lsq_solver(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            # accumulate Yt*Ci*p(i) in b
            b = np.zeros(factors)

            for j, cij in self._nonzeros(C, i):
                vj = Y[j]
                A += (cij - 1.0) * np.outer(vj, vj)
                b += cij * vj

            X[i] = np.linalg.solve(A, b)
        return X

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X

    def _nonzeros(self, R, row):
        for i in range(R.indptr[row], R.indptr[row + 1]):
            yield (R.indices[i], R.data[i])

    def _get_user_ratings(self, user_id):
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        return self.dataset[:, item_id]

    def filter_seen(self, playlist_id, scores):

        start_pos = self.dataset.indptr[playlist_id]
        end_pos = self.dataset.indptr[playlist_id + 1]

        playlist_profile = self.dataset.indices[start_pos:end_pos]

        scores[playlist_profile] = -np.inf

        return scores
