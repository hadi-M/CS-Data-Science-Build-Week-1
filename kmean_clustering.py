import pandas as pd

class MyKmeans():
    def __init__(self, k):
        self.k = k

    def initialize_centroids(X):
        return X.sample(self.k)

    def determine_clusters(X, centroid_list):
        df_only_features = X.copy()
        df_distance_columns = X.copy()
        for i in range(self.k):
            new_column_name = f"distance_from_cluster_{i}"
            centroid_list.iloc[i]
            centroid_copy_df = pd.DataFrame([centroid_list.iloc[i]] * self.k)
            df_only_features
            X[new_column_name] = 

    def fit(self, X):
        centroid_list = initialize_centroids(X)
        while True:
            new_clusters = determine_clusters(X, centroid_list)
            new_centroids = determine_new_centroids(new_clusters)
            if new_centroids == centroid_list:
                return new_centroids

            centroid_list = new_centroids




