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

    for playlist in targets.itertuples(index=True, name='Pandas'):
        playlist_id = getattr(playlist, 'playlist_id')

        recommended_items = recommender_object.recommend(playlist_id, at)
        print_to_file(playlist_id, recommended_items, file)

    file.close()
