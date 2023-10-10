import numpy as np

ratings_list = [i.strip().split("::") for i in open('.........ml-1m/ratings.dat').readlines()]
users_list = [i.strip().split("::") for i in open('..............ml-1m/users.dat').readlines()]
movies_list = [i.strip().split("::") for i in open('........ml-1m/movies.dat').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
ratings_df.head()

R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()
R = R_df.values
refined_dataset=ratings_df

#list of all users
unique_users = refined_dataset['UserID'].unique() 
#creating a list of all movie names in it
unique_movies = refined_dataset['MovieID'].unique()
len(unique_movies),len(unique_users)


users_list = refined_dataset['UserID'].tolist()
movie_list = refined_dataset['MovieID'].tolist()
len(users_list),len(movie_list)


ratings_list = refined_dataset['Rating'].tolist()
print(ratings_list)
len(ratings_list)

movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}
print(movies_dict)
print(len(movies_dict))


utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
print("Shape of Utility matrix: ",utility_matrix.shape)

for i in range(int(len(ratings_list))):

    utility_matrix[movies_dict[movie_list[i]]][int(users_list[i])-1] = ratings_list[i]

utility_matrix= utility_matrix.T