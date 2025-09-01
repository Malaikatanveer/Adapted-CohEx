import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max(**kw)[:, np.newaxis]), **kw)


class SRIDHCR:
    def __init__(self, n_clusters, beta=5.0):
        self.c = n_clusters
        self.beta = beta
        self.cluster_centers_ = None
        self.cluster_idx_ = None

    @staticmethod
    def assignment(x: np.ndarray, centroids: np.ndarray):
        print("inside assignment")
        # in case of feature are not 1-D (images), reshape
        x = x.reshape(x.shape[0], -1)
        centroids = centroids.reshape(centroids.shape[0], -1)

        dist = cdist(x, centroids)
        assignment = randargmax(-dist, axis=1)  # break ties randomly
        return assignment




    def penalty(self, x: np.ndarray, y: np.ndarray, centroids: np.ndarray):
        """
        Compute the penalty for a given set of centroids
        :param x: Features, shape (n, d).
        :param y: Labels, shape (n, m).
        :param centroids: Indices of centroids, shape (k,).
        :return:
        """
        n = x.shape[0]
        
        assignment = self.assignment(x, centroids)  # cluster assignments (may have gaps)
        unique_labels = np.unique(assignment)  # actual cluster labels (may skip some numbers)
        k = len(unique_labels)  # actual number of non-empty clusters
        
        # compute cluster-wise means correctly, even with missing cluster labels
        cy = np.array([
            np.mean(y[assignment == label], axis=0) for label in unique_labels
        ])
        
        # remap assignments to match indices of cy
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_assignment = np.array([label_to_index[label] for label in assignment])
        
        # compute penalty correctly
        error = np.sum((y - cy[mapped_assignment]) ** 2) / n
        penalty = error + self.beta * (np.sqrt((k - self.c) / n) if k > self.c else 0)
        
        return penalty



    def penalty_labels(self, x: np.ndarray, y: np.ndarray, labels: np.ndarray):
        n = x.shape[0]
        k = len(np.unique(labels))

        label_mapping = dict()
        for label in labels:
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
        labels = np.array([label_mapping[label] for label in labels])

        y = y.reshape(n, -1)

        cy = np.array([np.mean(y[labels == i], axis=0) for i in range(k)])
        error = np.sum((y - cy[labels]) ** 2) / n
        penalty = error + self.beta * (np.sqrt((k - self.c) / n) if k > self.c else 0)

        return penalty

    def fit(self, x: np.ndarray, y: np.ndarray, r=100):
        print("inside fit")
        n = x.shape[0]

        best_penalty = float('inf')
        best_idx = None  # best centroids idx overall

        for t in range(r):
            print("inside fit with for loop: ",t)
            k = np.random.randint(self.c + 1, 2 * self.c + 1)
            idx = np.random.choice(n, k, replace=False)  # best centroids idx in this iter
            penalty = self.penalty(x, y, x[idx])
            prev_penalty = float('inf')
            while penalty < prev_penalty:
                print("penalty < prev_penalty")
                prev_penalty = penalty

                # adding a centroid
                add_best_i = -1
                add_best_penalty = float('inf')
                for i in range(n):
                    print("inner for loop: ",i)
                    if i not in idx:
                        _idx = np.concatenate((idx, [i]))  # centroid idx to be tested
                        _penalty = self.penalty(x, y, x[_idx])
                        if _penalty < penalty:
                            add_best_i = i
                            add_best_penalty = _penalty

                # removing a centroid
                k = len(idx)
                remove_best_i = -1
                remove_best_penalty = float('inf')
                if k > 2:
                    for i in range(k):
                        _idx = np.delete(idx, i)
                        _penalty = self.penalty(x, y, x[_idx])
                        if _penalty < penalty:
                            remove_best_i = i
                            remove_best_penalty = _penalty

                if add_best_i != -1:
                    idx = np.concatenate((idx, [add_best_i]))
                    penalty = add_best_penalty
                if remove_best_i != -1 and remove_best_penalty < penalty:
                    idx = np.delete(idx, remove_best_i)
                    penalty = remove_best_penalty

            if penalty < best_penalty:
                best_penalty = penalty
                best_idx = idx

        self.cluster_idx_ = best_idx
        self.cluster_centers_ = x[best_idx]
        return self.cluster_centers_

    def predict(self, x: np.ndarray, _y: np.ndarray):
        print("predicting:")
        return self.assignment(x, self.cluster_centers_)

    def fit_predict(self, x: np.ndarray, y: np.ndarray, r=5):  
        print("inside fit_predict")
        self.fit(x, y, r)
        return self.predict(x, y)


def main():
    x = np.array([
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1],
        [0, 3],
        [1, 3],
        [2, 3],
        [5, 3],
        [6, 3],
        [7, 3],
        [0, 4],
        [1, 4],
        [2, 4],
        [5, 4],
        [6, 4],
        [7, 4],
        [2, 6],
        [3, 6],
        [4, 6],
        [5, 6],
        [2, 7],
        [3, 7],
        [4, 7],
        [5, 7]
    ])
    y = np.array([
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0
    ])

    cluster = SRIDHCR(n_clusters=2, beta=0.5)
    centroids = cluster.fit(x, y)
    centroid_idx = np.array([np.where(np.all(x == c, axis=1))[0][0] for c in centroids])
    print(centroids)
    print(centroid_idx)

    plt.figure(dpi=300)
    assignment = cluster.assignment(x, centroids)
    k = centroids.shape[0]
    for i in range(k):
        plt.scatter(centroids[i, 0], centroids[i, 1],
                    marker='o' if y[centroid_idx[i]] == 0 else 'x', color=plt.get_cmap('tab10')(i), s=200)
        plt.scatter(x[np.bitwise_and(y == 0, assignment == i), 0], x[np.bitwise_and(y == 0, assignment == i), 1],
                    marker='o', color=plt.get_cmap('tab10')(i))
        plt.scatter(x[np.bitwise_and(y == 1, assignment == i), 0], x[np.bitwise_and(y == 1, assignment == i), 1],
                    marker='x', color=plt.get_cmap('tab10')(i))
    plt.show()


if __name__ == '__main__':
    main()
