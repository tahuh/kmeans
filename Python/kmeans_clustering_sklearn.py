#!/usr/bin/python

from sklearn.cluster import KMeans

inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

engine = KMeans(n_clusters=3, init='random', max_iter = 1000)
kmeans = engine.fit(inputs)
print kmeans.labels_
print kmeans.cluster_centers_
