import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import time

from explainer import Explainer
from supervised_clustering import SRIDHCR


def cohex(explainer: Explainer, dataset: np.ndarray, n_cohorts: int, n_iter: int, termination_count: int,
          verbose=False, return_penalty=False, return_centroids=False):
    """
    Running CohEx.
    :param explainer: An `Explainer` instance. Should implement the
    `explain(dataset)` function as defined in `explainer.py`.
    :param dataset: np.ndarray, shape (n, f), where `n` is the number of
    samples, and `f` is the number of features.
    :param n_cohorts: _Expected_ number of cohorts.
    :param n_iter: Number of initializations to run iterative cohort
    explanation algorithms.
    :param termination_count: In each iteration, if the number of iteration
    without improvement exceeds this number, then the iteration would terminate.
    :param verbose: Whether to print debug information.
    :param return_penalty: If True, then the supervised clustering penalty is
    also returned. Incompatible with `return_centroids`.
    :param return_centroids: If True, then the centroids are also returned.
    Incompatible with `return_penalty`.
    :return k: int, the number of clusters.
    :return labels: np.ndarray, shape (n,), where each entry is an integer
    in [0, k).
    :return cohort_importance: np.ndarray, shape (k, f), the average importance
    of each cohort.
    """
    print("inside Cohex")
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values
    print("dataset: ",dataset.shape)
    n_instances = dataset.shape[0]
    importance = explainer.explain(dataset)
    clustering = SRIDHCR(n_clusters=n_cohorts)
    centroids = dataset[np.random.choice(n_instances, n_cohorts, replace=False)]
    labels = clustering.assignment(dataset, centroids)

    penalty_best = float('inf')
    centroids_best = None
    labels_best = None
    importance_best = None

    for t in range(n_iter):
        penalty_iter_best = float('inf')
        centroids_iter_best = None
        labels_iter_best = None
        importance_iter_best = None

        n_iter_no_improvement = 0
        i = 0
        while True:
            i += 1

            for j in range(centroids.shape[0]):
                indices = np.where(labels == j)[0]
                cohort = dataset[indices]
                cohort_values = explainer.explain(cohort)
                importance[indices] = cohort_values
            labels = clustering.fit_predict(dataset, importance)
            centroids = clustering.cluster_centers_

            penalty = clustering.penalty(dataset, importance, centroids)
            if verbose:
                print(f'iter {t}.{i}:\n\tpenalty={penalty:.6f}')

            if penalty < penalty_iter_best:
                print("penalty < penalty_iter_best")
                centroids_iter_best = centroids
                labels_iter_best = labels
                penalty_iter_best = penalty
                importance_iter_best = importance
                n_iter_no_improvement = 0
            else:
                print("penalty !< penalty_iter_best")
                n_iter_no_improvement += 1
                if n_iter_no_improvement >= termination_count:
                    break

        if penalty_iter_best < penalty_best:
            centroids_best = centroids_iter_best
            labels_best = labels_iter_best
            penalty_best = penalty_iter_best
            importance_best = importance_iter_best


    with open("cohex_iterations_log.txt", "a") as f:
        f.write(f"one full cohex done\n")
    # compute avg importance for each cohort
    print("Averge imporatnce now getting calculated")
    k = centroids_best.shape[0]
    shape = list(importance.shape)
    shape[0] = k
    cohort_importance = np.zeros(shape)
    for j in range(k):
        cohort_importance[j] = np.mean(importance_best[labels_best == j], axis=0)
    if return_penalty:
        return k, labels_best, cohort_importance, penalty_best
    elif return_centroids:
        return k, labels_best, cohort_importance, centroids_best
    else:
        return k, labels_best, cohort_importance





def kmeans_by_feature(explainer, dataset, n_cohorts):
    clustering = KMeans(n_clusters=n_cohorts, n_init='auto')
    importance = explainer.explain(dataset)

    labels = clustering.fit_predict(dataset.reshape(dataset.shape[0], -1))
    unique_labels = np.unique(labels)
    k_actual = len(unique_labels)  # number of non-empty clusters

    # prepare cohort_importance only for non-empty clusters
    shape = list(importance.shape)
    shape[0] = k_actual
    cohort_importance = np.zeros(shape)

    for idx, label in enumerate(unique_labels):
        mask = (labels == label)
        cohort_importance[idx] = np.mean(importance[mask], axis=0)

    # remap labels to contiguous range [0..k_actual-1]
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    labels_mapped = np.array([label_to_index[label] for label in labels])

    # get centroids (only those corresponding to non-empty clusters)
    raw_centroids = clustering.cluster_centers_
    centroids = np.array([raw_centroids[label] for label in unique_labels])

    return k_actual, labels_mapped, cohort_importance, centroids
