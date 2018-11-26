import pandas as pd
import scipy as sc
import scipy.sparse as sps

from sklearn.preprocessing import MultiLabelBinarizer


def create_matrices():
    # addresses of the files
    train_file = 'data/train.csv'
    target_playlists_file = 'data/target_playlists.csv'
    tracks_file = 'data/tracks.csv'

    # reading of all files and renaming columns
    train_data = pd.read_csv(train_file)
    train_data.columns = ['playlist_id', 'track_id']

    tracks_data = pd.read_csv(tracks_file)
    tracks_data.columns = ['track_id', 'album_id', 'artist_id', 'duration_sec']

    target_playlists = pd.read_csv(target_playlists_file)
    target_playlists.columns = ['playlist_id']

    # building the URM matrix
    grouped_playlists = train_data.groupby('playlist_id', as_index=True).apply(lambda x: list(x['track_id']))
    URM = MultiLabelBinarizer(sparse_output=True).fit_transform(grouped_playlists)
    URM_csr = URM.tocsr()

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

    # ignoring the 'duration' attribute
    '''
    durations = tracks_data.reindex(columns=['track_id', 'duration_sec'])
    durations.sort_values(by='track_id', inplace=True) # this seems not useful, values are already ordered
    durations_list = [[d] for d in durations['duration_sec']]
    icm_durations = MultiLabelBinarizer(sparse_output=True).fit_transform(durations_list)
    icm_durations_csr = icm_durations.tocsr()
    '''

    ICM = sc.sparse.hstack((icm_albums_csr, icm_artists_csr))
    ICM_csr = ICM.tocsr()

    return URM_csr, ICM_csr, target_playlists
