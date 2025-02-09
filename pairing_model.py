# pairing_model.py
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import defaultdict
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb+srv://dolientrang03:Dolientrang2003!@hackbeanpot.aemmt.mongodb.net/?retryWrites=true&w=majority&appName=Hackbeanpot')
db = client['test']
users_collection = db['users']

# Fetch data from MongoDB
users_cursor = users_collection.find()
users_data = list(users_cursor)

# Convert MongoDB data to DataFrame and ensure _id is a string
users_df = pd.DataFrame(users_data)
users_df['_id'] = users_df['_id'].astype(str)

# Ensure 'interests' column exists and handle missing values
if 'interests' in users_df.columns:
    users_df['interests'] = users_df['interests'].apply(lambda x: x if isinstance(x, list) else [])

    # Extract up to three interest categories
    for i in range(3):
        column_name = f'interest{i+1}'
        users_df[column_name] = users_df['interests'].apply(lambda x: x[i] if i < len(x) else 'None')
else:
    print("Warning: 'interests' column not found in database.")

# Ensure 'location' column exists
if 'location' not in users_df.columns:
    users_df['location'] = "Unknown"

# Define categorical features
categorical_features = ['location'] + [f'interest{i+1}' for i in range(3)]
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ]
)

# KNN Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', NearestNeighbors(n_neighbors=3, metric='euclidean'))
])

# Preprocess the data before fitting
processed_data = preprocessor.fit_transform(users_df)

# Fit the pipeline with processed data
pipeline.named_steps['knn'].fit(processed_data)

# Find nearest neighbors
neighbors = pipeline.named_steps['knn'].kneighbors(processed_data, return_distance=False)

# Function to pair users based on nearest neighbors
def pair_users_with_knn(df, neighbors):
    pairings = defaultdict(list)
    user_ids = df['_id'].tolist()
    
    for idx, user_id in enumerate(user_ids):
        for neighbor_idx in neighbors[idx]:
            if neighbor_idx != idx:  # Avoid self-pairing
                pairings[user_id].append(user_ids[neighbor_idx])
    
    return pairings

# Execute pairing function
pairings = pair_users_with_knn(users_df, neighbors)

# Function to get pairings
def get_pairings():
    return pairings



# # pairing_model.py
# import pandas as pd
# from pymongo import MongoClient
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.neighbors import NearestNeighbors
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from collections import defaultdict
# import random
# import numpy as np

# # Connect to MongoDB
# client = MongoClient('mongodb+srv://dolientrang03:Dolientrang2003!@hackbeanpot.aemmt.mongodb.net/?retryWrites=true&w=majority&appName=Hackbeanpot')
# db = client['test']
# users_collection = db['users']
# print(users_collection)

# # Fetch data from MongoDB
# users_cursor = users_collection.find()
# users_data = list(users_cursor)

# # Convert MongoDB data to DataFrame
# users_df = pd.DataFrame(users_data)

# # Merging the DataFrames on 'user_id'
# #match_users_df = pd.merge(match_users, users[['_id', 'interests', 'chatMatchedUsers']], left_on='userId', right_on='_id', how='left')

# # Extract five elements from the 'interests' column of matchUsers
# for i in range(3):
#     column_name = f'interest{i+1}'
#     users_df[column_name] = users_df['interests'].map(
#         lambda interests: interests[i]
#     )

# # Preprocessing
# interests_feature = [f'interest{i+1}' for i in range(3)]
# categorical_features = ['location'] + interests_feature
# categorical_transformer = OneHotEncoder()  # transform categorical features into a numerical format
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('columnTransformer', categorical_transformer, categorical_features)
#     ])

# # Create a pipeline that includes scaling and knn clustering
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('scaler', StandardScaler(with_mean=False)),
#     ('knn', NearestNeighbors(n_neighbors=3, metric='euclidean'))
# ])

# # Fit the pipeline to the user data
# pipeline.fit(users_df)

# # Apply KNN to find nearest neighbors
# knn_model = pipeline.named_steps['knn']
# neighbors = knn_model.kneighbors(users_df, return_distance=False)

# # Function to pair users based on nearest neighbors
# def pair_users_with_knn(df, neighbors):
#     pairings = defaultdict(list)
#     user_ids = df['_id'].astype(str).tolist()
    
#     for idx, user_id in enumerate(user_ids):
#         for neighbor_idx in neighbors[idx]:
#             if neighbor_idx != idx:  # Avoid self-pairing
#                 pairings[str(user_id)].append(str(user_ids[neighbor_idx]))
    
#     return pairings

# # Execute pairing function
# pairings = pair_users_with_knn(users_df, neighbors)

# # Function to get pairings
# def get_pairings():
#     return pairings