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


file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)
#
# file_path = os.path.expanduser('../ml-100k/smalltest.csv')
# reader = Reader(line_format='user item rating timestamp', sep=',')
# data1 = Dataset.load_from_file(file_path, reader=reader, rating_scale=(0, 5.0))
       
path = '../ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
id_list = df.id.values.tolist()
cat_df = df[['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
list_of_cats = cat_df.values.tolist()
taste_number_data = {}
# taste_score_data = {}
item_user_cate = {}
#
def into_rate(cate,rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate


class KNNWithMeans(SymmetricAlgo):


    def __init__(self,list_of_cats,taste_score_data, base_line=False, k=20, min_k=1, sim_options={}, verbose=True, **kwargs):

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

        # exit()
        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            # if user_based == False then:
                # nb = item_inner_id

            if sim > 0:
                # if self.sim_options.userbased == True:
                #     tt = i
                #     i = nb
                #     nb = tt
                # result = np.dot(self.list_of_cats[i],self.taste_score_data[str(trainset.to_raw_uid(nb))])


                # print('itemx_categories')
                # print(itemx_categories)
                # print('nb')
                # print(str(trainset.to_raw_uid(nb)))
                # print(taste_score_data[str(trainset.to_raw_uid(nb))])
                # print('result')
                # print(result)
                if self.base_line:
                    sum_sim += sim
                    sum_ratings += (r - self.means[nb]) * sim
                    actual_k += 1
                else:

                    result = np.dot(self.list_of_cats[nb], self.taste_score_data[str(trainset.to_raw_uid(y))])
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

        return est, details


k = 5

kf = KFold(n_splits=k, random_state=100)

t_mae = 0
t_rmse = 0

# trainset, testset = train_test_split(data1, test_size=.25, random_state=100)
for trainset, testset in kf.split(data1):
    taste_score_data = {}
    cat_score_data = {}
    item_cats = defaultdict(list)
    items_taste_score_data = {}
    #
    # # for trainset, testset in kf.split(data1):
    for user, item_list in trainset.ur.items():
        user_categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        blanklist = []
        # print('********' + str(user) + '*********')

        total_user_cater = 0
        raw_uid = trainset.to_raw_uid(user)
        # print(str(raw_uid))
        # print(item_list)
    #
        for item_inner_id,rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)
            # print("user " + user + ", " + raw_item_id + ", " + str(rating))
            # print("user " + raw_uid +", "+ raw_item_id + ", "+ str(rating))
            # print("movie:")
            # print("item_inner_id: " + raw_item_id)

    #         # print("item_inner_id: " + str(item_inner_id))
            itemx_categories = list_of_cats[int(raw_item_id)-1]
            # print(itemx_categories)
            itemx_ratings = copy.deepcopy(itemx_categories)
    #
            itemx_ratings = into_rate(itemx_ratings, rating)
            # print(itemx_ratings)


            item_cats[item_inner_id].append(itemx_ratings)
    #
            # user_categories = [a + b for a, b in zip(user_categories, itemx_categories)]
            user_rating = [a + b for a, b in zip(user_rating, itemx_ratings)]
            # print(user_rating)
            # set the taste_score_data


        user_rating = (user_rating / np.sum(user_rating)).tolist()

        taste_score_data[raw_uid] = user_rating

    for item, user_list in trainset.ir.items():
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for user, rating in user_list:
            temlist = [c*rating for c in taste_score_data[trainset.to_raw_uid(user)]]
            user_rating = [a + b for a, b in zip(temlist, user_rating)]

        itemx_rating = (user_rating/np.sum(user_rating)).tolist()
        # print(itemx_rating)
        items_taste_score_data[item] = itemx_rating
        # exit()
    # for key, cat_ratings in item_cats.items():
    #     cat_total_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     for item in cat_ratings:
    #         totalx_rating = [a + b for a, b in zip(cat_total_rating, item)]
    #
    #     items_taste_score_data[key] = [l/sum(cat_ratings) for l in totalx_rating]
    #     print('items_taste_score_data[key]')
    #     print(items_taste_score_data[key])


        # print(user_rating)
        # print('\n')


# print(len(taste_score_data))
    user_based = False  # changed to False to do item-absed CF
    sim_options = {'name': 'pearson', 'user_based': user_based}

    algo = KNNWithMeans(items_taste_score_data,taste_score_data,base_line=True, sim_options=sim_options)
    # algo = KNNWithMeans(sim_options=sim_options)

# Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

# Then compute RMSE
    t_rmse += accuracy.rmse(predictions)
    t_mae += accuracy.mae(predictions)



print("RMSE")
print(str(t_rmse/k))

print("MAE")
print(str(t_mae/k))

# pd.Dataframe(predictions, columns=['u_id', 'iid', 'rating', 'est', 'details'])

