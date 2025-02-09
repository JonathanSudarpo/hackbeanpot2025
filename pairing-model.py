# pairing_model.py
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import defaultdict
import random

# Connect to MongoDB
client = MongoClient('mongodb+srv://dolientrang03:<db_password>@hackbeanpot.aemmt.mongodb.net/?retryWrites=true&w=majority&appName=Hackbeanpot')
db = client['test']
users_collection = db['users']

# Fetch data from MongoDB
users_cursor = users_collection.find()
users_data = list(users_cursor)

# Convert MongoDB data to DataFrame
users_df = pd.DataFrame(users_data)

# Merging the DataFrames on 'user_id'
#match_users_df = pd.merge(match_users, users[['_id', 'interests', 'chatMatchedUsers']], left_on='userId', right_on='_id', how='left')

# Extract five elements from the 'interests' column of matchUsers
for i in range(5):
    column_name = f'interest{i+1}'
    users_df[column_name] = users_df['interests'].map(
        lambda interests: interests[i]
    )

# Preprocessing
interests_feature = [f'interest{i+1}' for i in range(5)]
categorical_features = ['location'] + interests_feature
categorical_transformer = OneHotEncoder()  # transform categorical features into a numerical format
preprocessor = ColumnTransformer(
    transformers=[
        ('columnTransformer', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes scaling and knn clustering
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', NearestNeighbors(n_neighbors=3, metric='euclidean'))
])

# Fit the pipeline to the user data
transformed_features = pipeline.fit_transform(users_df)

# Apply KNN to find nearest neighbors
knn_model = pipeline.named_steps['knn']
neighbors = knn_model.kneighbors(transformed_features, return_distance=False)

# Function to pair users based on nearest neighbors
def pair_users_with_knn(df, neighbors):
    pairings = defaultdict(list)
    user_ids = df['userId'].tolist()
    
    for idx, user_id in enumerate(user_ids):
        for neighbor_idx in neighbors[idx]:
            if neighbor_idx != idx:  # Avoid self-pairing
                pairings[str(user_id)].append(str(user_ids[neighbor_idx]))
    
    return pairings

# Execute pairing function
pairings = pair_users_with_knn(users_df, neighbors)

# Function to get pairings
def get_pairings():
    return pairings