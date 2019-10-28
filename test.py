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
#
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
taste_number_data = {}
item_user_cate = {}


#
def into_rate(cate, rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate


k = 2

kf = KFold(n_splits=k, random_state=100)

t_mae = 0
t_rmse = 0

for trainset, testset in kf.split(data1):
    taste_score_data = {}
    cat_score_data = {}
    item_cats = defaultdict(list)
    items_taste_score_data = {}

    for user, item_list in trainset.ur.items():

        user_categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        blanklist = []


        total_user_cater = 0

        for item_inner_id, rating in item_list:

            raw_item_id = trainset.to_raw_iid(item_inner_id)

            itemx_categories = list_of_cats[int(raw_item_id) - 1]

            itemx_ratings = copy.deepcopy(itemx_categories)

            itemx_ratings = into_rate(itemx_ratings, rating)

            item_cats[item_inner_id].append(itemx_ratings)

            user_rating = [a + b for a, b in zip(user_rating, itemx_ratings)]

        user_rating = (user_rating / np.sum(user_rating)).tolist()

        taste_score_data[user] = user_rating

    for item, user_list in trainset.ir.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for user, rating in user_list:
            temlist = [c * rating for c in taste_score_data[user]]
            user_rating = [a + b for a, b in zip(temlist, user_rating)]

        itemx_rating = (user_rating / np.sum(user_rating)).tolist()

        items_taste_score_data[item] = itemx_rating



