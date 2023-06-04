import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
# from constants import *
PATH_RECOMM_CLEAN= "data/processed/recommendation_clean_data.csv"
PATH_SENTIMENTS_CLEAN = "data/processed/sentiments_processed.csv"
PATH_VECTORIZER = "pickle/tfidf_vectorizer.pkl"
PATH_MODEL = "pickle/randomforest_model.pkl"
PRODUCT_COLUMN = "id"
USER_COLUMN = "reviews_username"
VALUE_COLUMN = "reviews_rating"


class RecommendationEngine:
    def __init__(self):
        self.reviews_df = pd.read_csv(PATH_RECOMM_CLEAN)
        self.sentiments = pd.read_csv(PATH_SENTIMENTS_CLEAN)
        self.tfidf_vectorizer = pickle.load(open(PATH_VECTORIZER, 'rb'))
        self.model = pickle.load(open(PATH_MODEL, 'rb'))
        
    def get_user_final_rating(self):
        df_recommendation = self.reviews_df[["id", "name", "reviews_rating", "reviews_username"]].copy()
        df_recommendation = df_recommendation[~df_recommendation['reviews_username'].isna()]
        train, test = train_test_split(df_recommendation, test_size=0.2, random_state=42)
        self.train = train
        df_pivot = pd.pivot_table(train,index=USER_COLUMN, columns = PRODUCT_COLUMN, values = VALUE_COLUMN).fillna(0)
        mean = np.nanmean(df_pivot, axis=1)
        df_subtracted = (df_pivot.T-mean).T
        dummy_train = train.copy()
        dummy_train[VALUE_COLUMN] = dummy_train[VALUE_COLUMN].apply(lambda x: 0 if x>=1 else 1)
        dummy_train = pd.pivot_table(dummy_train,index=USER_COLUMN, columns = PRODUCT_COLUMN, values = VALUE_COLUMN).fillna(1)
        user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
        user_correlation[np.isnan(user_correlation)] = 0
        user_correlation[user_correlation<0]=0
        user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
        user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
        return user_final_rating

    def get_recommendations_for_user(self, user_input, top_k = 20):
        user_final_rating = self.get_user_final_rating()
        recommendations = user_final_rating.loc[user_input].sort_values(ascending=False)[0:top_k]
        final_recommendations = pd.DataFrame({'product_id': recommendations.index, 'similarity_score' : recommendations})
        final_recommendations.reset_index(drop=True)
        final_df = pd.merge(final_recommendations, train, on="id")[["id", "name", "similarity_score"]].drop_duplicates()
        return final_df
    
    def get_recommendations_by_sentiment(self, user, top_k = 5):
        user_final_rating = self.get_user_final_rating()
        if (user in user_final_rating.index):
            recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            temp = self.sentiments[self.sentiments.id.isin(recommendations)]
            #temp["reviews_text_cleaned"] = temp["reviews_text"].apply(lambda x: self.preprocess_text(x))
            #transfor the input data using saved tf-idf vectorizer
            X =  self.tfidf_vectorizer.transform(temp["text_preprocessed"].values.astype(str))
            temp["Predicted Sentiment"]= self.model.predict(X)
            temp = temp[['name','Predicted Sentiment']]
            temp_grouped = temp.groupby('name', as_index=False).count()
            temp_grouped["Positive Review Count"] = temp_grouped.name.apply(lambda x: temp[(temp.name==x) & (temp['Predicted Sentiment']==1)]["Predicted Sentiment"].count())
            temp_grouped["Total Review Count"] = temp_grouped['Predicted Sentiment']
            temp_grouped['Percent Positive Reviews'] = np.round(temp_grouped["Positive Review Count"]/temp_grouped["Total Review Count"]*100,2)
            final = temp_grouped.sort_values('Percent Positive Reviews', ascending=False)
            print(final)
            return final
        else:
            print(f"User name {user} doesn't exist")
    