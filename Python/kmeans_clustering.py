#!/usr/bin/python

"""
kmeans_clustering.py

Performs k-means clustering using Expextation Maximization

Author : Sunghoon Heo

NO numpy used

Pure python code
"""
import random

class KMeansClusterEngine:
    """
    trainSet : Data to cluster
    N : Number of data
    dim : Dimension of vector
    k : K value ( number of clusters )
    iteration : How many times to perform EM
    row_major : is this input matrix is row major?
    """
    def __init__(self, trainSet, N, dim, k, iteration=1000, row_major=True):
        try:
            assert(len(trainSet[0]) == dim)
        except AssertionError:
            raise AssertionError("Input train set dimension and argument dimention do not match")
        try:
            assert(len(trainSet) == N)
        except AssertionError:
            raise AssertionError("Number of input training data set is different with parameter N")
        self.trainingSet = trainSet
        self.N = N ### rowsize
        self.dim = dim ### column size
        self.k = k
        self.iterations = iteration
        self.row_major= row_major
        if self.row_major == False:
            self.trainingSet = self.transpose(trainSet)
        self.solution = [] ### Our solution
        self.rnks = []
        self.label = [0] * self.N
        for n in range(self.N):
            self.rnks.append([0] * self.k )
        self.initialize()

    def transpose(self):
        mat = []
        for i in range(self.dim):
            mat.append([0] * self.N)
        for i in range(self.N):
            for j in range(self.dim):
                mat[j][i] = self.trainingSet[i][j]
        return mat
    def initialize(self):
        # pre selects k random centers
        self.solutions = random.sample(self.trainingSet, self.k)
    def distance(self ,v1,v2, n):
        dist = 0.0
        for i in range(n):
            dist += ((v1[i]-v2[i]) * (v1[i] - v2[i]))
        return dist
    def multiplyConstant(self, c , vector, n):
        v = []
        for i in range(n):
            v.append( c * vector[i] )
        return v
    def Estep(self):
        # Solves for Rnk
        for n in range(self.N):
            xn = self.trainingSet[n] # n-th data
            for k in range(self.k):
                distances = [] # Collection of distances
                for i, center_j in enumerate(self.solutions):
                    if i == k:
                        dist = float("inf") # always not consider this
                    dist = self.distance(xn, center_j, self.dim)
                    distances.append(dist)
                m = min(distances)
                argmin_j = distances.index(m)
                if argmin_j == k:
                    self.rnks[n][k] = 1
                else:
                    self.rnks[n][k] = 0
    def Mstep(self):
        uks = []
        for k in range(self.k):
            rnk_sum = 0
            uk = []
            for n in range(self.N):
                rnk_sum += self.rnks[n][k]
                xn = self.trainingSet[n]
                rnk_xn = self.multiplyConstant(self.rnks[n][k], xn, self.dim)
                if len(uk) == 0:
                    uk = rnk_xn
                else:
                    for i in range(self.dim):
                        uk[i] = uk[i] + rnk_xn[i]
            for i in range(self.dim):
                uk[i] = uk[i] / float(rnk_sum)
            uks.append(uk)
        self.solutions = uks
    def assign_labels(self):
        for n, xn in enumerate(self.trainingSet):
            distances = []
            for k in range(self.k):
                dist = self.distance(xn, self.solutions[k], self.dim)
                distances.append(dist)
            m = min(distances)
            argmin = distances.index(m)
            self.label[n] = argmin
    def solve(self):
        for i in range(self.iterations):
            self.Estep()
            self.Mstep()
            self.assign_labels()
    def show_solution(self):
        return self.solutions
    def show_labels(self):
        return self.label

if __name__ == "__main__":
# Driver program
    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    dim = 2
    N = len(inputs)
    k = 3
    engine = KMeansClusterEngine(inputs, N, dim, k)
    engine.solve()
    sol = engine.show_solution()
    for s in sol:
        print str(s)
    print engine.show_labels()
