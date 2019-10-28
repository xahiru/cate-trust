from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from surprise import Dataset
import pandas as pd
import numpy as np
from surprise.model_selection import KFold


def flat(listx):
    for item in listx:
        if not isinstance(item, (list, tuple)):
            yield item
        else:
            yield from flat(item)
def convert(listx):
    #l istx : ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']

    pass

file_path_save_data = 'data/processed/'  # don't forget to create this folder before running the scrypt
datasetname = 'ml-latest-small'  # valid datasetnames are 'ml-latest-small', 'ml-20m', and 'jester'
data2 = Dataset.load_builtin(datasetname)

path = '~/Desktop/RS/code/ml-latest-small/movies.csv'
df = pd.read_csv(path, sep=",", encoding="iso-8859-1", names=['id','name','CATE'])
df = df[['id','CATE']]
CATE = []
for cate in df['CATE']:
    # change cate(str) into the list of cate(list)
    cate = cate.split('|')
    print(cate)
    exit()
    # set a map function from cate(word) to cate(0or1)
    # cate = convert(cate)

    cate_temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # if 'Horror' in cate:
    #     cate_temp[1] = 1
    # if 'Horror' in cate:
    #     cate_temp[1] = 1
    # if 'Horror' in cate:
    #     cate_temp[1] = 1
    # if 'Horror' in cate:
    #         cate_temp[1] = 1
    # if 'Horror' in cate:
    #     cate_temp[1] = 1
    # if 'Horror' in cate:
    #     cate_temp[1] = 1
    # if 'Horror' in cate:
    #     cate_temp[1] = 1
print(set(list(flat(CATE))))

# cate = df['CATE'][1]
# print(type(cate))

# id_list = df.id.values.tolist()
# cat_df = df[['cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','cat19']]
# list_of_cats = cat_df.values.tolist()
# taste_data = {}

# kf = KFold(n_splits=2, random_state=100)
# for trainset, testset in kf.split(data2):
#     for user, item_list in trainset.ur.items():
#         user_categories = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         total_user_cater = 0
#         raw_uid = trainset.to_raw_uid(user)
#
#         for item_inner_id,_ in item_list:
#             raw_item_id = trainset.to_raw_iid(item_inner_id)
#             # print("movie:")
#             # print("raw_item_id: " + raw_item_id)
#             # print("item_inner_id: " + str(item_inner_id))
#             itemx_categories = list_of_cats[int(raw_item_id)-1]
#             # print(itemx_categories)
#
#             user_categories = [a + b for a, b in zip(user_categories, itemx_categories)]
#         movie_list = []
#         movie_list.append(user_categories)
#         user_categories = user_categories / np.sum(user_categories)
#         user_categories = user_categories.tolist()
#         movie_list.append(user_categories)
#         taste_data[raw_uid] = movie_list
#
# print(taste_data['590'])  # for example : 596 is the raw_user_id in data
# print(len(taste_data))


