import sys
sys.path.append("..........path_to_con_fusion")
import con_fusion as cf
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

import copy    

sel_data=copy.deepcopy(masked_train_data)    
p_users=.10
p_items=.3
#np.zeros((len(data), len(data[0])))#np.empty((int(len(data)*p_users), int(len(data[0])*p_items)))
rows=np.random.randint(0,len(sel_data), int(len(sel_data)*p_users))
cols=np.random.randint(0,len(sel_data[0]), int(len(sel_data[0])*p_items))
for i in rows:
    for j in cols:
        sel_data[i][j]=5


data_with_noise=sel_data


mf_faulty = MF_NEW(data_with_noise, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_faulty.train()
user_m2=mf_faulty.P
item_m2=mf_faulty.Q
result2=mf_faulty.full_matrix_new(user_m2, item_m2)

predicted_result2 = np.ma.masked_array(result2, test_mask) 

diff_test_noise= predicted_result2-(masked_test_data)
diff1_test_noise=np.abs(diff_test_noise)
RMSE_test_noise=np.sqrt(np.mean(np.square(diff_test_noise)))
print(RMSE_test_noise)
################
v1= data_with_noise- masked_train_data
v2=np.where(v1==0, np.nan, v1)
mask_v=np.isnan(v2)
v3=np.ma.masked_array(data_with_noise, mask_v)
item_avg=np.zeros(len(masked_train_data[0]))
for i in range(0, len(masked_train_data[0])-1):
    item_avg[i]=np.mean(masked_train_data[:,i])
    if np.isnan(item_avg[i])== True:
        item_avg[i]=1

###########################################  

v4=copy.deepcopy(v3)
for i in range(0, len(v3)):
    for j in range(0, len(v3[0])):
        # if v3[i,j] ==np.nan:            
        if v3[i,j]>0:
            v4[i,j]=item_avg[j]
             

mf_interm = MF_NEW(v4, K=64, alpha=0.03, beta=0.01, iterations=150)
mf_interm.train()

user_m3=mf_interm.P
item_m3=mf_interm.Q
result3=mf_interm.full_matrix_new(user_m3, item_m3)

predicted_result3 = np.ma.masked_array(result3, test_mask) 
diff_test_intm= predicted_result3-(masked_test_data)
diff1_test_intm=np.abs(diff_test_intm)
RMSE_test_intm=np.sqrt(np.mean(np.square(diff_test_intm)))
print(RMSE_test_intm)

net =cf.ConFusion(64,64)
a1=torch.tensor(user_m2).float()
a2=torch.tensor(user_m3).float()
y_1= net(a1,a2,"user")


a11=torch.tensor(item_m2).float()
a21=torch.tensor(item_m3).float()
y_2= net(a11,a21,"item")

y_11=y_1.detach().numpy()
y_12=y_2.detach().numpy()

res=mf_faulty.full_matrix_new(y_11,y_12)
predicted_result_conv = np.ma.masked_array(res, test_mask) 

diff_test_conv= predicted_result_conv-(masked_test_data)
print("Hello! this is your RMSE after conv_transfer")
RMSE_test_conv=np.sqrt(np.mean(np.square(diff_test_conv)))
print(RMSE_test_conv)
print(f"Noisy model RMSE {RMSE_test_noise} RMSE")
print(f"final model RMSE {RMSE_test_conv} RMSE")
print(f"Intermediate model RMSE {RMSE_test_intm} RMSE")