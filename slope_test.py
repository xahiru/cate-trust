from surprise import SlopeOne
from surprise import Dataset
from surprise.model_selection import cross_validate
import numpy as np  # noqa
import numpy as np
from six.moves import range
from six import iteritems

from surprise.prediction_algorithms.predictions import PredictionImpossible
from surprise.prediction_algorithms.algo_base import AlgoBase
import heapq
from surprise.model_selection import KFold
import pandas as pd
from surprise import accuracy
import copy


np.seterr(divide='ignore',invalid='ignore')


# Load the movielens-100k dataset (download it if needed),


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


def dist(vec1, vec2, common_number):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)) / common_number)
    return dist


def common_number(vec1, vec2):
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
# We'll use the famous SVD algorithm.




class SlopeONE(AlgoBase):


    def __init__(self,option_dev):

        AlgoBase.__init__(self)
        self.option_dev = option_dev

    def fit(self, trainset):

        n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|

        # u, i, j, r_ui, r_uj

        AlgoBase.fit(self, trainset)

        freq = np.zeros((trainset.n_items, trainset.n_items), np.int)
        dev = np.zeros((trainset.n_items, trainset.n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    freq[i, j] += 1
                    dev[i, j] += r_ui - r_uj


        for i in range(n_items):
            dev[i, i] = 0
            for j in range(i + 1, n_items):

                dev[i, j] /= freq[i, j]
                dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

        # mean ratings of all users: mu_u
        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        Ri = [j for (j, _) in self.trainset.ur[u] if self.freq[i, j] > 0]
        est = self.user_mean[u]
        
        if Ri:
            est += sum(self.option_dev[i][j] for j in Ri) / len(Ri)

        return est

# algo = SlopeONE()

k = 5
kf = KFold(n_splits=k, random_state=100)
t_mae = 0
t_rmse = 0

for trainset, testset in kf.split(data1):
    taste_score_data = {}
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
    # sim_taste_np = np.array(sim_taste)




    # sim_taste_np = None


#---------------------------------main part---------------------------------#


    algo = SlopeONE(sim_taste)

    algo.fit(trainset)
    predictions = algo.test(testset)

# Then compute RMSE
    t_rmse += accuracy.rmse(predictions)
    t_mae += accuracy.mae(predictions)

#---------------------------------show result---------------------------------#

print("\nMEAN_RMSE:" + str(t_rmse/k))
print("MEAN_MAE:" + str(t_mae/k))