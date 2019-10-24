from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import Dataset
import pandas as pd
import numpy as np
from surprise.model_selection import KFold
import copy


file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-100k'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data1 = Dataset.load_builtin(datasetname)

path = '~/Desktop/RS/code/ml-100k/u.item'
df = pd.read_csv(path, sep="|", encoding="iso-8859-1", names=['id','name','date','space','url','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19'])
id_list = df.id.values.tolist()
cat_df = df[['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
list_of_cats = cat_df.values.tolist()
taste_number_data = {}
taste_score_data = {}
item_user_cate = {}

def into_rate(cate,rate):
    for index in range(len(cate)):
        if cate[index] == 1:
            cate[index] = rate
    return cate



kf = KFold(n_splits=2, random_state=100)
for trainset, testset in kf.split(data1):


    for user, item_list in trainset.ur.items():
        user_categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        user_rating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        total_user_cater = 0
        raw_uid = trainset.to_raw_uid(user)

        for item_inner_id,rating in item_list:
            raw_item_id = trainset.to_raw_iid(item_inner_id)
            # print("movie:")
            # print("raw_item_id: " + raw_item_id)
            # print("item_inner_id: " + str(item_inner_id))
            itemx_categories = list_of_cats[int(raw_item_id)-1]
            # print(itemx_categories)
            itemx_ratings = copy.deepcopy(itemx_categories)

            itemx_ratings = into_rate(itemx_ratings,rating)



            user_categories = [a + b for a, b in zip(user_categories, itemx_categories)]
            user_rating = [a + b for a, b in zip(user_rating, itemx_ratings)]


        # set the taste_number_data
        movie_list = []
        movie_list.append(user_categories)
        user_categories = user_categories / np.sum(user_categories)
        user_categories = user_categories.tolist()
        movie_list.append(user_categories)
        taste_number_data[raw_uid] = movie_list

        # set the taste_score_data
        _ = []
        _.append(user_rating)
        numb = len(item_list)
        user_rating = [item / numb for item in user_rating]
        # user_rating = user_rating.tolist()
        _.append(user_rating)
        taste_score_data[raw_uid] = _



    for item, user_list in trainset.ir.items():
        Sum_taste = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        raw_item_id = trainset.to_raw_iid(item) # type:str
        # item_user_cate =
        for user_inner_id, rating in user_list:
            user_raw_id = trainset.to_raw_uid(user_inner_id)
            taste_one_user = taste_score_data[str(user_raw_id)][1]
            Sum_taste = [a + b for a, b in zip(taste_one_user, Sum_taste)]


        item_user_cate[raw_item_id] = Sum_taste
        # get the user's for this item

             # for each user

             # get this user's taste

             # sum

        # calc the item's user cate

        # make a dict to save it

# print(taste_number_data['590'][1])  # for example : 596 is the raw_user_id in data
# print(taste_score_data['590'][1])  # for example : 596 is the raw_user_id in data
# print(item_user_cate['318'])
# print(list_of_cats[int('208') - 1])
# print(len(taste_number_data))
# print(list_of_cats)

# print(taste_score_data['1458'][1])
print(len(taste_score_data))



