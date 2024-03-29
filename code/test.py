from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import Dataset
import pandas as pd
import numpy as np
from surprise.model_selection import KFold
import numpy as np
from six import iteritems
import heapq
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.predictions import PredictionImpossible
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.knns import SymmetricAlgo
import copy
from collections import defaultdict
import os
import matplotlib
import plotly

file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)

# file_path = os.path.expanduser('../ml-100k/smalltest.csv')
# reader = Reader(line_format='user item rating timestamp', sep=',')
# data1 = Dataset.load_from_file(file_path, reader=reader, rating_scale=(0, 5.0))

path = '../ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1",
                 names=['id', 'name', 'date', 'space', 'url', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7',
                        'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18',
                        'cat19'])
id_list = df.id.values.tolist()
cat_df = df[['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13',
             'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19']]
list_of_cats = cat_df.values.tolist()


def into_rate(cate, rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate

def dist(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

def common_number(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    i = 0
    common_set_length = 0
    while (i < len(vec1)):
        a_val = vec1[i]
        b_val = vec2[i]

        if a_val != 0 and b_val != 0:
            common_set_length += 1
        i += 1

    return common_set_length

k = 2
kf = KFold(n_splits=k, random_state=100)
t_mae = 0
t_rmse = 0

for trainset, testset in kf.split(data1):
    taste_score_data = {}
    items_taste_score_data = {}
    sim_taste = []

    # calc each user's taste (by this user's history item list)
    for user, item_list in trainset.ur.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for item_inner_id, rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)

            itemx_cate = copy.deepcopy(list_of_cats[int(raw_item_id) - 1])

            itemx_rating = into_rate(itemx_cate, rating)

            user_rating = [a + b for a, b in zip(user_rating, itemx_rating)]

        # get the proportion of each cate

        user_rating = (user_rating / np.sum(user_rating)).tolist()
        taste_score_data[user] = user_rating

    #calc each item's cate (by its used user' rating and user's taste)
    for item, user_list in trainset.ir.items():
        user_ratings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for user, rating in user_list:
            weighted_rating = [taste * rating for taste in taste_score_data[user]]
            user_ratings = [a + b for a, b in zip(weighted_rating, user_ratings)]

        # get the proportion of each cate
        itemx_rating = (user_ratings / np.sum(user_ratings)).tolist()
        items_taste_score_data[item] = itemx_rating
# ---------------------------------
#TODO list:
    # 1.make the sim matrix between user by taste (UxU)
    # 2.make the result matrix by cate and taste (UxI)
    # 3.make the sim matrix between item by cate (IxI)

#compute the result matrix by cate and taste (UxI)


#compute the sim between users

    # for i in range(len(taste_score_data)):
    #
    #     sim_taste_x = []
    #     user_x_taste = taste_score_data[i]
    #
    #     for j in range(len(taste_score_data)):
    #
    #         user_y_taste = taste_score_data[j]
    #
    #         #compute the common taste
    #         common_taste = common_number(user_x_taste,user_y_taste)
    #
    #         #compute the dist between two users
    #         dist_temp = dist(user_x_taste, user_y_taste)
    #         sim_taste_x.append(dist_temp/common_taste)
    #
    #     sim_taste.append(sim_taste_x)
    # sim_taste_np = np.array(sim_taste)
    # print(len(taste_score_data))
    # print(len(items_taste_score_data))
    # for i in range(len(items_taste_score_data)):
    #     pass
    # print(i)
    #
    #
    #
    #
    #
    # print(type(sim_taste))
    # sim_taste_np = np.array(sim_taste)
    # print(type(sim_taste_np))

    # print(sim_taste[2][1])
    # exit()




#problem list : when I tried to implement the code about Item's sim, for each different round, the size of items_taste_score_data is different






# print(predictions)

# df = pd.DataFrame(predictions, columns=['u_id', 'i_id', 'rating', 'estimate','detail'])

# print(df)
