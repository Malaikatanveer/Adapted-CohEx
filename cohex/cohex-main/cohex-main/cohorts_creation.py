import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tqdm import trange
from cohort_explanation import cohex,kmeans_by_feature
from scenario.CNA import GeneScenario
from explainer import SHAPExplainer
from eval import locality, stability_importance
from supervised_clustering import SRIDHCR
import argparse

#============================ For CNA-based survival Z-scores=================================#
#-----------------------Script to find cohorts and imporatnce using optimal k--------------------------#





def cohorts_creation(method,  n_cohorts=12): 
    """
    Generate and save cohorts and their explanations for the gene dataset using SHAP.

    This function clusters samples into cohorts using the specified method 
    (`cohex` or `kmeans_by_feature`) and computes SHAP-based feature importances 
    for each cohort. The cohorts and their explanations are saved to disk.

    Parameters
    ----------
    method : str
        Clustering method to use. Must be one of:
        - "cohex": Use the CohEx clustering algorithm.
        - "kmeans_by_feature": Use K-means clustering based on feature importances.
    n_cohorts : int, optional, default=12
        Number of cohorts (clusters) to form.

    Saves
    -----
    The following files are saved in the `output/gene_shap` directory:
    - `{method}_shap_centroids.npy`: Centroids for each cohort.
    - `{method}_shap_labels.npy`: Cluster label for each sample.
    - `{method}_shap_importance.npy`: SHAP importance values for each cohort.

    Notes
    -----
    - SHAP is used to compute cohort-level feature importances.
    - The `output/gene_shap` directory is created automatically if it does not exist.
    - Cohort information and their explanations are also printed to the console.
    """
    scenario = GeneScenario()
    save_dir='output/gene_shap'
    print("directory created")
    os.makedirs(save_dir, exist_ok=True)


    print("Explainer started")
    explainer = SHAPExplainer(scenario.model.predict, mode='default')
    print("Explainer ended")


    if method == 'kmeans_by_feature':
        k, labels, importance, centroids = kmeans_by_feature(explainer, scenario.X, n_cohorts=n_cohorts)
        print("kmeans_by_feature ended")
        np.save(os.path.join(save_dir, f'{method}_shap_centroids.npy'), centroids)
        print("saved centroids")
    elif method == 'cohex':
        print("starting CohEx")
        k, labels, importance, centroids = cohex(explainer, scenario.X, n_cohorts=n_cohorts,
                                                 n_iter=5, termination_count=5, verbose=False, return_centroids=True)
        print("CohEx ended")
        np.save(os.path.join(save_dir, f'{method}_shap_centroids.npy'), centroids)
        print("saved centroids")
    else:
        raise ValueError
    np.save(os.path.join(save_dir, f'{method}_shap_labels.npy'), labels)
    np.save(os.path.join(save_dir, f'{method}_shap_importance.npy'), importance)

    for i in range(k):
        print(f'cohort {i}')
        print('importance:')
        print(importance[i])



def main():
    parser = argparse.ArgumentParser(description="Run cohorts_creation with configurable cohorts.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["cohex", "kmeans_by_feature"],
        help="Clustering method to use."
    )
    parser.add_argument(
        "--n_cohorts",
        type=int,
        default=12,
        help="Number of cohorts to form (default: 12). Specifiy the optimal no. of cohorts"
    )
    args = parser.parse_args()

    print(f"Running cohorts_creation with method={args.method}, n_cohorts={args.n_cohorts}")
    cohorts_creation(method=args.method, n_cohorts=args.n_cohorts)


if __name__ == "__main__":
    main()  