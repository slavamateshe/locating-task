import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

DIST_COEF = 0.1
DESCENT_COEF = 1.2
def euclidian_distance(x1, y1, x2, y2): # calculates euclidian distance between 2 points
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)

def mean_distance(df): # calculates mean distance between points of transmitter
    df_len = df.shape[0]
    dist = 0
    for i in range(df_len):
        for j in range(df_len):
            dist += euclidian_distance(df['transmitter_lat'][i], df['transmitter_lon'][i],
                                       df['transmitter_lat'][j], df['transmitter_lon'][j])
    return dist / (df_len ** 2)

def center_radius(df):
    samples = df.shape[0]
    max_lbl = -1      # index of most common cluster in cluster labelling
    X = df[['transmitter_lat', 'transmitter_lon']]
    distance = mean_distance(df) * DIST_COEF
    cl = DBSCAN(eps=distance, min_samples=samples)
    while max_lbl == -1:
        cl = DBSCAN(eps=distance, min_samples=samples).fit(X)
        values, counts = np.unique(cl.labels_, return_counts=True)
        ind = np.argmax(counts)
        max_lbl = values[ind]
        samples = int(samples / DESCENT_COEF)
    x = df["transmitter_lat"][cl.labels_ == max_lbl].mean()
    y = df["transmitter_lon"][cl.labels_ == max_lbl].mean()
    r = 0
    for i in range(len(cl.labels_)):
        if cl.labels_[i] == max_lbl and geodesic((x, y), (df["transmitter_lat"][i],
                                                          df["transmitter_lon"][i])).meters > r:
            r = geodesic((x, y), (df["transmitter_lat"][i], df["transmitter_lon"][i])).meters
    return x, y, r

def main():
    print("write path to dataset")
    path = input().strip()
    df = pd.read_csv(path, on_bad_lines='skip')[["transmitter_lat", "transmitter_lon"]].dropna()
    x, y, r = center_radius(df)
    print("lat", x)
    print("lon", y)
    print("radius", r, "meters")

main()





