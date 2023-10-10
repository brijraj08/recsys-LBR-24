import numpy as np
import pandas as pd
from MF_AI import MF_NEW

# =============================================================================
column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('................/ml-100k/u.data', sep='\t',header=None,names=column_names1)
dataset.head() 
d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
print(column_names2)
items_dataset = pd.read_csv('.............../ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
items_dataset
movie_dataset = items_dataset[['movie id','movie title']]
movie_dataset.head()


merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
merged_dataset.head()

refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
refined_dataset.head()


#list of all users
unique_users = refined_dataset['user id'].unique() 
#creating a list of all movie names in it
unique_movies = refined_dataset['movie title'].unique()
len(unique_movies),len(unique_users)


users_list = refined_dataset['user id'].tolist()
movie_list = refined_dataset['movie title'].tolist()
len(users_list),len(movie_list)


ratings_list = refined_dataset['rating'].tolist()
print(ratings_list)
len(ratings_list)

movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}
print(movies_dict)
print(len(movies_dict))

## creating a utility matrix for the available data
utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
print("Shape of Utility matrix: ",utility_matrix.shape)

for i in range(len(ratings_list)):

  utility_matrix[movies_dict[movie_list[i]]][users_list[i]-1] = ratings_list[i]

utility_matrix= utility_matrix.T

#Masking
mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)

#creating a test matrix of 20%
test_m=np.zeros((len(masked_arr), len(masked_arr[0])))


for i in range(0, len(masked_arr)):
    for j in range(0, len(masked_arr[0])):
        if masked_arr.mask[i][j]==False:
            r1=np.random.randint(1,100,1)
            if r1 <20:
                test_m[i][j]=masked_arr[i][j]


#replacing 0 with nan
test_nan = np.where(test_m==0, np.nan, test_m)
#creating mask for test dataset matrix
test_mask = np.isnan(test_nan)          
masked_test_data = np.ma.masked_array(test_nan, test_mask) 
train_nan= utility_matrix-test_m
train_mask = np.isnan(train_nan)
masked_train_data = np.ma.masked_array(train_nan, train_mask)

mf_orig = MF_NEW(masked_train_data, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_orig.train()
user_m1=mf_orig.P
item_m1=mf_orig.Q
result1=mf_orig.full_matrix_new(user_m1, item_m1)

predicted_result1 = np.ma.masked_array(result1, test_mask) 

diff_test= predicted_result1-(masked_test_data)
print("RMSE_test")
RMSE_test=np.sqrt(np.mean(np.square(diff_test)))
print(RMSE_test)