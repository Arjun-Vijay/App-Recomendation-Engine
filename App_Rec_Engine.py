import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# read the csv files using the appropriate file path
app_data = pd.read_csv('AppData.csv', encoding='latin1')
app_descriptions = pd.read_csv('appleStore_description.csv', encoding='latin1')


# combine product descriptions with product category to analyze all text-based info
total_app_descriptions = app_data['app_desc']
genre = app_data['prime_genre']
total_app_descriptions.append(genre)
app_data['word_data'] = total_app_descriptions

# construct a TF-IDF Matrix to compute similarity scores between products
tf = TfidfVectorizer(input='word', encoding='utf-8', stop_words='english')
tfid_matrix = tf.fit_transform(app_data['word_data'])
similarity_scores = linear_kernel(tfid_matrix, tfid_matrix)


results = {}

# Orders the similarity scores in descending order and store the corresponding app names
for index, row in app_data.iterrows():
    similarity_indices = similarity_scores[index].argsort()[:-100:-1]
    similiar_items = [(similarity_scores[index][i], app_data['track_name'][i]) for i in similarity_indices]
    results[row['id']] = similiar_items[1:]


# get the name of a given app
def getItemName(id):
    return app_data.loc[app_data['id'] == id]['track_name'].tolist()[0].split(' - ')[0]

# Prints out the recomendations


def recommend(item_id, num):
    print('Recommending {0} products similar to {1}:'.format(num, getItemName(item_id)))
    print("----------------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + rec[1] + " (similarity score: " + str(rec[0]) + ")")


# Item we would like to base recomendation of off.
recommend(item_id=1, num=5)
