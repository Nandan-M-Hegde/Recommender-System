import numpy as np
import pandas as pd

class Popularity_Recommender:
    def __init__(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        self.recommendations = None
        return

    def Create(self):
        train_data_grouped = self.train_data.groupby([self.item_id]).agg({self.user_id : 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id':'score'}, inplace=True)

        train_data_sort = train_data_grouped.sort_values(["score", self.item_id], ascending=[0,1])
        train_data_sort["Rank"] = train_data_sort["score"].rank(ascending=0, method="first")

        self.recommendations = train_data_sort.head(10)
        return

    def Recommend(self, user_id):
        user_recommendations = self.recommendations
        user_recommendations['user_id'] = user_id

        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations

class Similarity_Recommender:
    def __init__(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.recommendations = None
        return

    def Get_User(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user = list(user_data[self.item_id].unique())
        return user

    def Get_Item(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item = set(item_data[self.user_id].unique())
        return item

    def Get_Unique_Data(self):
        return list(self.train_data[self.item_id].unique())

    def Construct_Cooccurrence_Matrix(self, user_songs, all_songs):
        users = []

        for a_song in user_songs:
            users.append(self.Get_Item(a_song))

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(0, len(all_songs)):
            songs_i = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                users_j = users[j]
                users_intersection = users_i.intersection(users_j)
                
                if len(users_intersection) != 0:
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
        
        self.cooccurence_matrix = cooccurence_matrix
        return

    def Generate_Top_Recommendations(self, user, user_songs, all_songs):
        print("Non-zero values in cooccurence_matrix :%d" % np.count_nonzero(self.cooccurence_matrix))
        
        user_sim_scores = self.cooccurence_matrix.sum(axis=0)/float(self.cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        df = pd.DataFrame(columns=['user_id', 'song', 'score', 'rank'])
         
        rank = 1 
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
            
        return df

    def Recommend(self, user):
        user_songs = self.Get_User(user)
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        all_songs = self.Get_Unique_Data()
        print("No. of unique songs in the training set: %d" % len(all_songs))
       
        self.Construct_Cooccurrence_Matrix(user_songs, all_songs)
        return self.Generate_Top_Recommendations(user, user_songs, all_songs)
