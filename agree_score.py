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

datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)

# file_path = os.path.expanduser('../ml-100k/smalltest.csv')
# reader = Reader(line_format='user item rating timestamp', sep=',')
# data1 = Dataset.load_from_file(file_path, reader=reader, rating_scale=(0, 5.0))
       
path = '../ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
id_list = df.id.values.tolist()
cat_df = df[['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
list_of_cats = cat_df.values.tolist()

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


class KNNWithMeans(SymmetricAlgo):


    def __init__(self,list_of_cats,taste_score_data,option_sim,base_line=False, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):


        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k
        self.list_of_cats = list_of_cats
        self.taste_score_data = taste_score_data
        self.base_line = base_line
        self.option_sim = option_sim



    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        ir_backup = copy.deepcopy(self.trainset.ur)
        ur_backup = copy.deepcopy(self.trainset.ir)


        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        my_ur = defaultdict(list)
        my_ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r in self.trainset.all_ratings():
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            result = np.dot(self.list_of_cats[iid], self.taste_score_data[uid])
            my_ur[uid].append((iid, result))
            my_ir[iid].append((uid, result))




        self.sim = self.compute_similarities(verbose=self.verbose)

        # self.sim = self.
        self.trainset.ur = ur_backup
        self.trainset.ir = ir_backup


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



k = 5
kf = KFold(n_splits=k, random_state=100)
t_mae = 0
t_rmse = 0

for trainset, testset in kf.split(data1):
    taste_score_data = {}
    items_taste_score_data = {}
    sim_taste = []
#compute each user's taste (by this user's history item list)
    for user, item_list in trainset.ur.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for item_inner_id,rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)

            itemx_cate = copy.deepcopy(list_of_cats[int(raw_item_id)-1])

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

#compute the sim between users

    for i in range(len(taste_score_data)):

        sim_taste_x = []
        user_x_taste = taste_score_data[i]

        for j in range(len(taste_score_data)):

            user_y_taste = taste_score_data[j]

            #compute the common taste
            common_taste = common_number(user_x_taste,user_y_taste)

            #compute the dist between two users
            dist_temp = dist(user_x_taste, user_y_taste,common_taste)
            sim_taste_x.append(dist_temp)

        sim_taste.append(sim_taste_x)
    sim_taste_np = np.array(sim_taste)
    # sim_taste_np = None


#---------------------------------main part---------------------------------#

    user_based = True  # changed to False to do item-absed CF
    sim_options = {'name': 'pearson', 'user_based': user_based}

    algo = KNNWithMeans(items_taste_score_data,taste_score_data,sim_taste_np,base_line=False, sim_options=sim_options)

    algo.fit(trainset)
    predictions = algo.test(testset)

# Then compute RMSE
    t_rmse += accuracy.rmse(predictions)
    t_mae += accuracy.mae(predictions)

#---------------------------------show result---------------------------------#

print("\nMEAN_RMSE:" + str(t_rmse/k))
print("MEAN_MAE:" + str(t_mae/k))

# print(predictions)

# df = pd.DataFrame(predictions, columns=['u_id', 'i_id', 'rating', 'estimate','detail'])

# print(df)
