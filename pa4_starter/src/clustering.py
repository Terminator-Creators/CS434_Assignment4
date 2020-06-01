import numpy as np
import random
import math
import statistics as st

class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    # DONE ------
    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    # DONE -------
    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """
        print("initing center")
        # Using a for loop, we select random seeds that arent the same to be our clusters
        centers = np.zeros(self.k, dtype=list)
        
        for i in range(self.k):
            rand = random.randint(0, len(x)-1)
            
            if(np.any(centers[:] == x[rand])):
                i -= 1
            else:
                centers[i] = x[rand]

        self.centers = centers

    # HUH?? 
    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """
        print("revising center")
        # I think this updates it?
        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)        
            self.centers[i] = x[wherei, :].mean(0)

    # DONE ------
    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        print("predicting...")
        labels = np.zeros((x.shape[0]), dtype=int)
        
        for i, line in enumerate(x):
            # Each line is an example of 500ish features
            smalldist = 100000000
            center = -1
            for j in range(len(self.centers)):
                dist = abs(self.distanceCalc(line, self.centers[j]))
                if(dist < smalldist):
                    smalldist = dist
                    center = j
            labels[i] = center

        return labels

    # DONE ------
    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """
        print("calculating SSE")
        sse = 0.
        for i in range(len(labels)):
            dist = self.distanceCalc(x[i], self.centers[labels[i]])
            # Undo the sqrt that the distance formula does
            sse += (dist * dist)

        return sse

    # DONE ------
    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        print("calculating purity")
        # Using 1-indexing to work with the output script
        labels = self.predict(x)
        labels = [label + 1 for label in labels]
        purity = [[]]*self.k
        kPurities = [0]*self.k
        for k in range(self.k):
            for i in range(len(labels)):
                if(y[i] == k+1):
                    purity[k].append(labels[i])

            # Gotta get the mode of each subpurity, then divide that number by the size of the buket
            mode = st.mode(purity[k])
            kPurities[k] = purity[k].count(mode)/len(purity[k])
        return st.mean(kPurities)

    # DONE ------
    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """
        print("Fitting to the dataset")
        # intialize self.centers
        self.init_center(x)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter

    # DONE ------
    def distanceCalc(self, x, y):
        """
        this function is a utility to compute the distance between two points
        :param x: point x (row of 500ish features)
        :param y: center point (row of 500ish features)
        :return: computes distance between the point and the center.
        """
        # Start assuming the points are the same, calc the differences squared and sum them
        # After the sum, we can sqrt it
        dist = 0
        print(x)
        print(y)
        for i in range(len(x)):
            dist += ((x[i] - y[i]) * (x[i] - y[i]))

        return math.sqrt(dist)