import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import Recommenders as Recommenders

class Recommender:
    def __init__(self):
        self.db = None
        self.song_grouped = None
        self.users = None
        self.songs = None
        self.test_data = None
        self.train_data = None
        self.Popularity_Model = None
        return

    def Load_Data(self):
        db1 = pd.read_csv("10000.txt", sep='\t', header=None, names=["user_id", "song_id", "listen_count"])
        db2 = pd.read_csv("song_data.csv")
        self.db = pd.merge(db1, db2.drop_duplicates(["song_id"]), on="song_id", how="left")
        return

    def Transform_Data(self):
        self.db = self.db.head(10000)
        self.db['song'] = self.db['title'].map(str) + " - " + self.db['artist_name']
        self.song_grouped = self.db.groupby(["song"]).agg({"listen_count": "count"}).reset_index()
        grouped_sum = self.song_grouped["listen_count"].sum()
        self.song_grouped['percentage']  = self.song_grouped["listen_count"].div(grouped_sum)*100
        self.song_grouped.sort_values(["listen_count", "song"], ascending = [0,1])
        return

    def Find_Unique(self):
        self.users = self.db["user_id"].unique()
        self.songs = self.db["song"].unique()
        return
    
    def Split_Data(self):
        self.train_data, self.test_data = train_test_split(self.db, test_size = 0.20, random_state=0)
        return

    def Create_Model(self):
        self.Popularity_Model = Recommenders.Popularity_Recommender(self.train_data, "user_id", "song")
        self.Popularity_Model.Create()

    def Get_Popular_Recommendations(self, user_id):
        if user_id<1 and user_id>=len(self.users):
            print("Enter valid user id")
            return
        
        return self.Popularity_Model.Recommend(user_id)

    def Get_Similar_Recommendations(self, user_id):
        self.Similarity_Model = Recommenders.Similarity_Recommender(self.train_data, "user_id", "song")
        
        user_items = self.Similarity_Model.Get_User(user_id)
        print("------------------------------------------------------------------------------------")
        print("Training data songs for the user userid: %s:" % user_id)
        print("------------------------------------------------------------------------------------")

        for user_item in user_items:
            print(user_item)

        print("----------------------------------------------------------------------")
        print("Recommendation process going on:")
        print("----------------------------------------------------------------------")
        print(self.Similarity_Model.Recommend(self.users[user_id]))

R = Recommender()
R.Load_Data()
R.Transform_Data()
R.Find_Unique()
R.Split_Data()
R.Create_Model()
R.Get_Similar_Recommendations(5)