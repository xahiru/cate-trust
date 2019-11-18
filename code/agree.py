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
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.predictions import PredictionImpossible
from surprise.prediction_algorithms.algo_base import AlgoBase


file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)

path = '~/Desktop/RS/code/ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
id_list = df.id.values.tolist()
cat_df = df[['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
list_of_cats = cat_df.values.tolist()
taste_data = {}

kf = KFold(n_splits=2, random_state=100)
for trainset, testset in kf.split(data1):
    for user, item_list in trainset.ur.items():
        user_categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        total_user_cater = 0
        raw_uid = trainset.to_raw_uid(user)

        for item_inner_id,_ in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)
            # print("movie:")
            # print("raw_item_id: " + raw_item_id)
            # print("item_inner_id: " + str(item_inner_id))
            itemx_categories = list_of_cats[int(raw_item_id)-1]
            # print(itemx_categories)
            user_categories = [a + b for a, b in zip(user_categories, itemx_categories)]
        movie_list = []
        movie_list.append(user_categories)
        user_categories = user_categories / np.sum(user_categories)
        user_categories = user_categories.tolist()
        movie_list.append(user_categories)
        taste_data[raw_uid] = movie_list

class SymmetricAlgo(AlgoBase):

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):


        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class KNNWithMeans(SymmetricAlgo):


    def __init__(self, k=40, min_k=1, sim_options={'name': 'pearson'}, verbose=True, **kwargs):
        # 'trainset': trainset,
        # 'beta': 2.5,
        # 'epsilon': 1

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k

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

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        xitemcate = list_of_cats[int(i) - 1]


        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                # use this nbr id to get the taste of this user
                if nb == 0:
                    pass
                else:
                    print(xitemcate)
                    print(taste_data[str(nb)][1])
                    result = np.dot(xitemcate, taste_data[str(nb)][1])
                    print(result)
                # get the results : sum all the taste if this taste is the cate of this item

                # multipy the result with 'sim'
                    sum_sim += sim
                    print(sim)
                    sum_ratings += (r - self.means[nb]) * (((sim**2 + result**2)/2)**0.5)
                    actual_k += 1


        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details



# print(taste_data['596'])  # for example : 596 is the raw_user_id in data
# print(len(taste_data))

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)


algo = KNNWithMeans()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)
