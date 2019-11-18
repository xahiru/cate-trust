from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from surprise import AlgoBase
from surprise.model_selection import KFold
import multiprocessing
from surprise import Dataset
from surprise import KNNWithMeans
from surprise.accuracy import rmse
from surprise.accuracy import mae
import time
from surprise.cate_agreement import agree_trust


file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-latest-small'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data = Dataset.load_builtin(datasetname)

class AgreeTrustAlgorithm(AlgoBase):

    def __init__(self, k=40, min_k=1, alog=KNNWithMeans, user_based=True, beta=2.5, epsilon=0.9, lambdak=0.9,
                 sim_options={}, verbose=True, **kwargs):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)
        self.k = k  # todo :what's meaning of k here
        self.min_k = min_k
        self.algo = alog(k=k, sim_options=sim_options, verbose=True)
        self.epsilon = epsilon
        self.lambdak = lambdak
        self.beta = beta
        if user_based:
            self.ptype = 'user'
        else:
            self.ptype = 'item'

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        self.algo.fit(trainset)

        print('Ignore the above similiary matrix generation message, its not used in this algorithm')

        # tr = agree_trust(trainset, self.beta, self.epsilon, self.algo.sim, ptype=self.ptype, istrainset=True, activity=False)
        # self.algo.sim = tr**self.lambdak - (self.epsilon*noncom)
        print(self.algo.sim)
        print(self.algo.sim.shape)
        return self

    def estimate(self, u, i):



        return self.algo.estimate(u, i)

kf = KFold(n_splits=5, random_state=100)


sum_rmse = 0
sum_mae = 0
kt = 0
user_based = True  # changed to False to do item-based CF
epsilon = 1
lambdak = 11000
beta = 2.5



for trainset, testset in kf.split(data):

    sim_options = {'name': 'agreement', 'trainset': trainset, 'beta': beta, 'epsilon': epsilon}

    predict_alog = KNNWithMeans
    algo = AgreeTrustAlgorithm(k=40, alog=predict_alog, user_based=user_based, beta=beta, epsilon=epsilon,
                               lambdak=lambdak, sim_options=sim_options, verbose=True)
    algo_name = 'AgreeTrustAlgorithm'
    # # # #     # train and test algorithm.
    start = time.time()
    algo.fit(trainset)
    print(time.time() - start)

    start = time.time()
    predictions = algo.test(testset)
    print(time.time() - start)

    # # # Compute and print RMSE and MAE
    m_rmse = rmse(predictions, verbose=False)
    sum_rmse += m_rmse
    m_mae = mae(predictions, verbose=False)
    sum_mae += m_mae
    kt += 1
    print('m_rmse')
    print(m_rmse)
    print('m_mae')
    print(m_mae)
    print(datasetname)

mean_mae = sum_mae / kt
mean_rmse = sum_rmse / kt
if algo_name == 'AgreeTrustAlgorithm':
    print(algo_name + '_predict_alog_' + str(predict_alog) + '_user_based_' + str(user_based) + '_epsilon_' + str(
        epsilon) + '_lambdak_' + str(lambdak))
elif algo_name == 'OdnovanAlgorithm':
    print(algo_name + '_predict_alog_' + str(predict_alog) + '_user_based_' + str(user_based) + '_alpha_' + str(alpha))
else:
    print(algo_name + '_user_based_' + str(user_based))
print('mean_rmse')
print(mean_rmse)
print('mean_mae')
print(mean_mae)
 