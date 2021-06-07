from sklearn_extra.cluster import KMedoids
import numpy as np
import gower

def Kmenoids(X,K,maxIter):
    print("Fitting KMenoids")
    X_gower = gower.gower_matrix(X)
    kmedoids = KMedoids(n_clusters=K,metric="precomputed",max_iter=maxIter).fit(X_gower)
    return kmedoids.predict(X_gower)
