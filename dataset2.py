from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import Dataset
import pandas as pd
import numpy as np
from surprise.model_selection import KFold
from surprise import Dataset
from surprise.model_selection import KFold
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

# util part
def flat(listx):
    for item in listx:
        if not isinstance(item, (list, tuple)):
            yield item
        else:
            yield from flat(item)
def convert(listx):
    #listx : ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']
    list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if '(no genres listed)' in listx:
        list[0] = 1
    if 'Action' in listx:
        list[1] = 1
    if 'Adventure' in listx:
        list[2] = 1
    if 'Animation' in listx:
        list[3] = 1
    if 'Children' in listx:
        list[4] = 1
    if 'Comedy' in listx:
        list[5] = 1
    if 'Crime' in listx:
        list[6] = 1
    if 'Documentary' in listx:
        list[7] = 1
    if 'Drama' in listx:
        list[8] = 1
    if 'Fantasy' in listx:
        list[9] = 1
    if 'Film-Noir' in listx:
        list[10] = 1
    if 'Horror' in listx:
        list[11] = 1
    if 'Musical' in listx:
        list[12] = 1
    if 'Mystery' in listx:
        list[13] = 1
    if 'Romance' in listx:
        list[14] = 1
    if 'Sci-Fi' in listx:
        list[15] = 1
    if 'Thriller' in listx:
        list[16] = 1
    if 'War' in listx:
        list[17] = 1
    if 'Western' in listx:
        list[18] = 1
    return list
def into_rate(cate,rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate
def dist(vec1,vec2,common_number):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(np.sum(np.square(vec1 - vec2))/common_number)
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

#------------------------------------------------------------------------
# TODO list:
# get the 'list_of_cats' in the ml-latest-small dataset as option_dataset_1
# get the 'list_of_cats' in the ml-1M dataset as option_dataset_2
    # first need to add 1M to surprise framework
#------------------------------------------------------------------------

file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-latest-small'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)

path = '../ml-latest-small/movies.csv'
df = pd.read_csv(path, sep=",", encoding="iso-8859-1", names=['id','name','CATE'])
df = df[['id','CATE']]
list_of_cats = {}

for row in df.itertuples(index=True, name='Pandas'):

    id = str(getattr(row, "id"))

    cate = getattr(row, "CATE")

    # change cate(str) into the list of cate(list)
    cate = cate.split('|')

    # set a map function from cate(word) to cate(0or1)
    cate = convert(cate)

    list_of_cats[id] = cate

class KNNWithMeans(SymmetricAlgo):


    def __init__(self,list_of_cats,taste_score_data,base_line=False, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):


        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.list_of_cats = list_of_cats
        self.taste_score_data = taste_score_data
        self.base_line = base_line



    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.sim = self.compute_similarities(verbose=self.verbose)

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)
        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])
        # if user_based == False then:
            # x = i
            # y = u

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            side_info = {}
            # if user_based == False then:
                # nb = item_inner_id
            if sim > 0:

                if self.base_line:
                    # sim += 0.1005
                    sum_sim += sim
                    sum_ratings += (r - self.means[nb]) * sim
                    actual_k += 1
                else:
                    if self.sim_options['user_based'] == True:
                        result = np.dot(self.list_of_cats[i], self.taste_score_data[nb])
                    else:
                        result = np.dot(self.list_of_cats[nb], self.taste_score_data[y]) # y is the user in the item_based

                    # result += 0.3
                    sum_sim += result
                    sum_ratings += (r - self.means[nb]) * result
                    actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim

        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        #add list: error, result, user's taste ,movie's cate
        return est, details




t_mae = 0
t_rmse = 0
k = 5
kf = KFold(n_splits=k, random_state=100)

for trainset, testset in kf.split(data1):
    taste_score_data = {}
    items_taste_score_data = {}
    sim_taste = []
#compute each user's taste (by this user's history item list)
    for user, item_list in trainset.ur.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for item_inner_id,rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)

            itemx_cate = copy.deepcopy(list_of_cats[raw_item_id])

            itemx_rating = into_rate(itemx_cate, rating)

            user_rating = [a + b for a, b in zip(user_rating, itemx_rating)]

    #get the proportion of each cate

        user_rating = (user_rating / np.sum(user_rating)).tolist()
        taste_score_data[user] = user_rating

#compute each item's cate (by its used user' rating and user's taste)
    for item, user_list in trainset.ir.items():
        user_ratings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for user, rating in user_list:
            weighted_rating = [taste * rating for taste in taste_score_data[user]]
            user_ratings = [a + b for a, b in zip(weighted_rating, user_ratings)]

    # get the proportion of each cate
        itemx_rating = (user_ratings/np.sum(user_ratings)).tolist()
        items_taste_score_data[item] = itemx_rating


#---------------------------------main part---------------------------------#

    user_based = True  # changed to False to do item-absed CF

    #for normal case-------------------------------------------------------#
    sim_options = {'name': 'pearson', 'user_based': user_based}
    algo = KNNWithMeans(items_taste_score_data,
                        taste_score_data,
                        base_line=False,
                        sim_options=sim_options)

    #for agreement case
    # epsilon = 1
    # lambdak = 0.5
    # beta = 2.5
    # sim_options = {'name': 'agreement',
    #                'user_based': user_based,
    #                'trainset':trainset,
    #                'beta':beta,
    #                'epsilon':epsilon,
    #                'lambdak':lambdak}
    #
    # algo = KNNWithMeans(items_taste_score_data,
    #                     taste_score_data,
    #                     base_line=False,
    #                     sim_options=sim_options)

    algo.fit(trainset)
    predictions = algo.test(testset)

# Then compute RMSE
    t_rmse += accuracy.rmse(predictions)
    t_mae += accuracy.mae(predictions)

#---------------------------------show result---------------------------------#

print("\nMEAN_RMSE:" + str(t_rmse/k))
print("MEAN_MAE:" + str(t_mae/k))

# print(list_of_cats['193565'])
# print(len(list_of_cats))
# exit()








