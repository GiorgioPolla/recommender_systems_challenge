## Overview
This repository contains the code we came up for the Kaggle competition of the Recommender Systems course at Politecnico di Milano.

The team is made up by:
- **Polla Giorgio**
- **Romeo Paolo**

The application domain is a music streaming service, where users listen to tracks (songs) and create playlists of favorite songs. The main goal of the competition is to discover which track a user will likely add to a playlist, therefore "continuing" the playlist.
This is realized recommending a list of 10 relevant tracks for each target playlist.

## Data
The dataset includes around 1.2M interactions (tracks belonging to a playlist) with 50k playlists and 20k items (tracks).
A subset of 10k playlists has been selected as target playlists.
Target playlists are divided into two groups of 5k playlists each:
- Playlists in the first group are sequential: for these playlists, tracks in the training set are provided in sequential order;
- Playlists in the second group are random: for these playlists, tracks in the training set are provided in random order.

In Dataset/Data, you can find the following files:
- train.csv : the training set describing which tracks are included in the playlists;
- tracks.csv : supplementary information about tracks;
- target_playlists.csv : the set of target playlists that will receive recommendations;
- train_sequential.csv: list of ordered interactions for the 5k sequential playlists.

## Evaluation
As said before, 10k playlists are defined as target. The recommendations made for the target playlists are evaluated using the Mean Average Precision at 10 (MAP@10).

## Structure of the code
Here is a brief description of the structure of the code. 
Packages:
- Recommenders - contains all the algorithm implemented or just adapted from existing source code (mainly from the public repository of the course);
- Dataset - contains the datasets and a class useful to read and manipulate them;
- Utilities - contains supplementary classes and functions;
- Results - contains some files used to keep trace of the local evaluations;

The previous packages contain the necessary to build up the following scripts:
- test_single - used to evaluate the performance of single algorithms;
- test_hybrid - used to evaluate the performance of the hybrid algorithm;
- tune_hybrid - used to tune the weights to assign to the single algorithms in the hybrid solution;
- tune_single - used to tune the parameters of single algorithms;
- make_recommendations - used to generate the file in the correct format for the submission on Kaggle.

## Best solution

The best solution comes from the hybrid algorithm made up by the mix of the following approaches:
- Content based;
- Item based collaborative filtering;
- User based collaborative filtering;
- Graph based RP3Beta;
- Slim BPR;
- Matrix factorization ALS.

Firstly, every single algorithm has been tuned either manually or using the library Skopt (https://scikit-optimize.github.io/) and then, using the best hyperparameters for each algorithm, the **hybrid** has been realized by mixing the scores provided by each of them. 
Skopt has been used to determine the **weights** to assign to single algorithms.

Last but not least, during the building of the URM matrix has been taken into account the **importance of the sequentiality** of the 5k ordered target playlists. More importance has been assigned to the last tracks of each playlists following the principle that a user is more willing to replace the less important tracks with new ones, hence the last tracks could be more likely to be replaced with similar ones.

## Requirements
To be able to run the code, the following modules are required:
- scikit-learn;
- numpy
- scipy;
- pandas;
- tqdm.

## Acknowledgements
Some complex algorithms are taken and adapted from a github folder provided by our instructors.
