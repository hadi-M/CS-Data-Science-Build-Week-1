import pandas as pd
from ipdb import set_trace as st


class MyKmeans():
    def __init__(self, k, random_state=30, n_init=5):
        self.k = k
        self.random_state = random_state
        self.centroids = None
        self.n_init = n_init

    def initialize_centroids(self, X):
        return X.sample(self.k).reset_index(drop=True)

    def determine_clusters(self, X, centroid_list):
        df_only_features = X.copy()
        df_with_distance_columns = X.copy()
        for i in range(self.k):
            new_column_name = f"distance_from_cluster_{i}"
            row_count = len(df_with_distance_columns)
            centroid_copy_df = pd.DataFrame([centroid_list.iloc[i]] * row_count).reset_index(drop=True) 
            df_with_distance_columns[new_column_name] = ((centroid_copy_df-X)**2).sum(axis=1)**0.5

        labels = df_with_distance_columns.iloc[:, 2:].idxmin(axis=1).str.split("_", expand=True).iloc[:, -1]
        labels = labels.rename("labels").astype(int)
        clusters_with_labels = pd.concat([df_only_features, labels], axis=1)
        sum_of_distance_from_each_cluster = df_with_distance_columns.iloc[:, -self.k:].sum().sum()
        # st()
        return clusters_with_labels, sum_of_distance_from_each_cluster

    def determine_new_centroids(self, cluster_list):
        return cluster_list.groupby(["labels"]).mean()

    def single_fit(self, X):
        centroid_list = self.initialize_centroids(X)
        clusters = None
        while True:
            new_clusters, sum_of_distance = self.determine_clusters(X, centroid_list)
            new_centroids = self.determine_new_centroids(new_clusters)
            if new_clusters.equals(clusters):
                return centroid_list, sum_of_distance

            clusters = new_clusters
            centroid_list = new_centroids

    def fit(self, X):
        clusteriods_and_distances = []
        for i in range(self.n_init):
            # st()
            print(f"DONE WITH {i}")
            clusteriods_and_distances.append(self.single_fit(X))
        self.centroids = min(clusteriods_and_distances, key=lambda x: x[1])[0]

    def predict(self, X):
        # st()
        return self.determine_clusters(X, self.centroids)[0]["labels"]


if __name__ == "__main__":
    df = pd.read_csv("./kmean_data.csv")
    model = MyKmeans(4, n_init=40)
    # y = model.single_fit(df)
    model.fit(df)
    y = model.predict(df)
    st()