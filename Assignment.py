from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch, AgglomerativeClustering
import numpy as np

def AssignmentFunction(X,K,maxIter):
    '''
    Returns the cluster assigment of 2d data relative to multiple algorithms
        Parameters:
            X[2xN Matrix]: 2d numpy array of data
            K[int]: Number of clusters
        Returns:
            3xN dataset of cluster Assignment (3 Clustering Algorithms)
    '''
    def kMeans(X):
        print("Fitting Kmeans")
        def calcSqDistances(X, Kmus):
            # returns n x K matrix of distances where each row is a data pt and the columns are distances to K center
            distances = np.zeros((len(X), len(Kmus)))
            for i in range(len(X)):
                for z in range(len(Kmus)):
                    distances[i][z] = np.linalg.norm(X[i]-Kmus[z])
            return distances

        def determineRnk(sqDmat):
            # one-hot encode what cluster the data pt belongs to n x k
            Rnk = np.zeros((len(sqDmat), len(sqDmat[0])))
            for i in range(len(sqDmat)):
                Rnk[i][np.argmin(sqDmat[i])] = 1

            return Rnk

        def recalcMus(X, Rnk):
            # return a matrix of K rows, 2 cols
#             numClusters = len(Rnk[0])
#             newMus = np.zeros((numClusters, 2))

#             for i in range(numClusters):
#                 num = 0
#                 avg = np.zeros((1,2))
#                 for z in range(len(X)):
#                     if Rnk[z][i] == 1:
#                         avg = avg + X[z]
#                         num +=1
#                 avg = avg / num
#                 newMus[i]=avg
            R = np.argmax(Rnk, axis=1)
            newMus = np.zeros((np.shape(Rnk)[1], np.shape(X)[1]))
        
            for r in set(R):
                newMus[r] = np.mean(X[R == r, :], axis=0)

            return newMus

        # Determine and store data set information
        N = np.shape(X)[0]
        D = np.shape(X)[1]

        # Allocate space for the K mu vectors
        Kmus = np.zeros((K, D))

        # Initialize cluster centers by randomly picking points from the data
        rndinds = np.random.permutation(N)
        Kmus = X[rndinds[:K]];

        for iter in range(maxIter):
            sqDmat = calcSqDistances(X, Kmus);

            Rnk = determineRnk(sqDmat)

            KmusOld = Kmus

            Kmus = recalcMus(X, Rnk)
            
            print("iter: ", iter)
            # Check to see if the cluster centers have converged.  If so, break.
            if sum(abs(KmusOld.flatten() - Kmus.flatten())) < 1e-6:
                print("Converged!")
                break
        return Rnk

    def MOG():
        print("Fitting MOG")
        gm = GaussianMixture(n_components=K,max_iter=maxIter).fit(X)
        return gm.predict(X)

    def birch():
        print("Fitting Birch")
        b = Birch(n_clusters=K).fit(X)
        return b.predict(X)


    kMeansResults = np.argmax(kMeans(X),axis=1)
    mogResults = MOG()
    birchResults = birch()
    return np.array((kMeansResults, mogResults, birchResults))
