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

np.seterr(divide='ignore',invalid='ignore')


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.




class SlopeONE(AlgoBase):


    def __init__(self):

        AlgoBase.__init__(self)

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
            est += sum(self.dev[i, j] for j in Ri) / len(Ri)

        return est

algo = SlopeONE()

# Run 5-fold cross-va lidation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)