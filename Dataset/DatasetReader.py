import scipy.sparse as sps
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from Utilities.Support import split_train_validation_test


class DatasetReader(object):

    def __init__(self, split_train_test_validation_quota=[0.6, 0.2, 0.2], modelNumber="1"):

        if sum(split_train_test_validation_quota) != 1.0 or len(split_train_test_validation_quota) != 3:
            raise ValueError(
                "DatasetReader: splitTrainTestValidation must be a probability distribution over Train, Test and Validation")

        print("DatasetReader: loading data...")

        dataSubfolder = "./Dataset/"

        try:
            self.URM = sps.load_npz(dataSubfolder + "URM.npz")
            self.URM_train = sps.load_npz(dataSubfolder + "URM_train_" + modelNumber + ".npz")
            self.URM_test = sps.load_npz(dataSubfolder + "URM_test_" + modelNumber + ".npz")
            self.URM_validation = sps.load_npz(dataSubfolder + "URM_validation_" + modelNumber + ".npz")
            self.ICM = sps.load_npz(dataSubfolder + "ICM.npz")
            self.target_playlists = self.load_target_playlists()
            self.sequential_playlists = self.load_sequential_playlists()

        except FileNotFoundError:

            print("DatasetReader: URM_train or URM_test or URM_validation not found. Building new ones")

            self.load_matrices()

            self.URM_train, self.URM_test, self.URM_validation = split_train_validation_test(self.URM,
                                                                                             split_train_test_validation_quota)

            print("DatasetReader: saving URM_train and URM_test")
            sps.save_npz(dataSubfolder + "URM_train_" + modelNumber + ".npz", self.URM_train)
            sps.save_npz(dataSubfolder + "URM_test_" + modelNumber + ".npz", self.URM_test)
            sps.save_npz(dataSubfolder + "URM_validation_" + modelNumber + ".npz", self.URM_validation)
            sps.save_npz(dataSubfolder + "ICM.npz", self.ICM)

        print("DatasetReader: loading complete")

    def get_URM(self):
        return self.URM

    def get_URM_train(self):
        return self.URM_train

    def get_URM_test(self):
        return self.URM_test

    def get_URM_validation(self):
        return self.URM_validation

    def get_ICM(self):
        return self.ICM

    def get_target_playlists(self):
        return self.target_playlists

    def get_sequential_playlists(self):
        return self.sequential_playlists

    def load_target_playlists(self):
        target_playlists_file = './Dataset/Data/target_playlists.csv'
        target_playlists = pd.read_csv(target_playlists_file)
        target_playlists.columns = ['playlist_id']
        return target_playlists["playlist_id"].unique()

    def load_sequential_playlists(self):
        sequential_train_file = './Dataset/Data/train_sequential.csv'
        sequential_data = pd.read_csv(sequential_train_file)
        sequential_data.columns = ['playlist_id', 'track_id']
        return sequential_data['playlist_id'].unique()

    def load_matrices(self):

        # addresses of the files
        train_file = './Dataset/Data/train.csv'
        target_playlists_file = './Dataset/Data/target_playlists.csv'
        tracks_file = './Dataset/Data/tracks.csv'
        sequential_train_file = './Dataset/Data/train_sequential.csv'

        train_data = pd.read_csv(train_file)
        tracks_data = pd.read_csv(tracks_file)
        sequential_data = pd.read_csv(sequential_train_file)
        target_data = pd.read_csv(target_playlists_file)

        # building the URM taking into account the order of the 5k target playlists
        sequential_playlists = sequential_data['playlist_id'].unique()
        target_playlists = target_data['playlist_id'].unique()
        all_playlists = train_data['playlist_id'].unique()

        t = train_data.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
        t = t.drop(sequential_playlists)

        s = sequential_data.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))

        n_playlists = all_playlists.size
        n_tracks = tracks_data['track_id'].unique().size

        first_iteration = True

        for i in tqdm(range(n_playlists)):

            if i in s:
                # generate an array of weights for the playlist
                data = []
                indices = s[i]
                n = len(indices)

                # top is the variable value
                top = 0.93
                base = 1 - top
                weight = top / n

                index = 0
                incr = weight

                while index < n:
                    data.append(base + incr)
                    incr += weight
                    index += 1

                # assign the right weight to the right track
                index = 0
                d = {}

                while index < n:
                    d[indices[index]] = data[index]
                    index += 1

                # build the row, transform to csr and concatenate to the matrix
                row = np.zeros(n_tracks)

                for key in sorted(d.keys()):
                    row[key] = d[key]
                if first_iteration:
                    mat = sps.csr_matrix(row)
                    first_iteration = False
                else:
                    row_csr = sps.csr_matrix(row)
                    mat = sps.vstack((mat, row_csr))

            else:
                indices = t[i]

                row = np.zeros(n_tracks)

                for j in range(len(indices)):
                    row[indices[j]] = 1.0
                if first_iteration:
                    mat = sps.csr_matrix(row)
                    first_iteration = False
                else:
                    row_csr = sps.csr_matrix(row)
                    mat = sps.vstack((mat, row_csr))

        self.URM = mat
        sps.save_npz("./Dataset/URM.npz", mat)

        # building the ICM matrix
        artists = tracks_data.reindex(columns=['track_id', 'artist_id'])
        artists.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        artists_list = [[a] for a in artists['artist_id']]
        icm_artists = MultiLabelBinarizer(sparse_output=True).fit_transform(artists_list)
        icm_artists_csr = icm_artists.tocsr()

        albums = tracks_data.reindex(columns=['track_id', 'album_id'])
        albums.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        albums_list = [[a] for a in albums['album_id']]
        icm_albums = MultiLabelBinarizer(sparse_output=True).fit_transform(albums_list)
        icm_albums_csr = icm_albums.tocsr()

        durations = tracks_data.reindex(columns=['track_id', 'duration_sec'])
        durations.sort_values(by='track_id', inplace=True)  # this seems not useful, values are already ordered
        durations_list = [[d] for d in durations['duration_sec']]
        icm_durations = MultiLabelBinarizer(sparse_output=True).fit_transform(durations_list)
        icm_durations_csr = icm_durations.tocsr()

        ICM = sps.hstack((icm_albums_csr, icm_artists_csr, icm_durations_csr))
        self.ICM = ICM.tocsr()

        self.target_playlists = target_playlists
