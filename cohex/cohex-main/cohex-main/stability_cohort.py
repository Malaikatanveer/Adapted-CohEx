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
#----------------------Script to find cohort stability using optimal k---------------------------------#


def stability_cohort(method, n_iter=5, n_cohorts=12):
    """
    Assess the stability of gene cohorts across multiple iterations using Adjusted Rand Score (ARS).

    Cohorts are generated with the specified method ('cohex' or 'kmeans_by_feature'), 
    and stability is measured by comparing cohort labels across `n_iter` runs. 

    Parameters
    ----------
    method : str
        Clustering method: 'cohex' or 'kmeans_by_feature'.
    n_iter : int, optional (default=5)
        The number of iterations to repeat cohort generation and 
        cohort stability estimation.
    n_cohorts : int, optional (default=12)
        Number of cohorts to generate per iteration.

    Raises
    ------
    ValueError
        If an unrecognised method is provided.

    Output
    ------
    Writes results to a (.txt) file of format ***_shap_cohort_stability.txt
    in output/gene_shap folder

    """
    print("No of total iterations: ",n_iter)
    scenario = GeneScenario()
    save_dir='output/gene_shap'
    os.makedirs(save_dir, exist_ok=True)
    explainer = SHAPExplainer(scenario.model.predict, mode='default')

    if method == 'kmeans_by_feature':
        _, labels, _, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=n_cohorts)
    elif method == 'cohex':
        _, labels, _ = cohex(explainer, scenario.X, n_cohorts=n_cohorts, n_iter=5, termination_count=5, verbose=False)
    else:
        raise ValueError
    losses = np.zeros(n_iter)

    for i in trange(n_iter):
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
        if method == 'kmeans_by_feature':
            _, labels_alt, _, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=n_cohorts)
        elif method == 'cohex':
            _, labels_alt, _ = cohex(explainer, scenario.X, n_cohorts=n_cohorts, n_iter=5, termination_count=5, verbose=False)
        else:
            raise ValueError
        losses[i] = adjusted_rand_score(labels, labels_alt)

    with open(os.path.join(save_dir, f'{method}_shap_cohort_stability.txt'), 'w') as f:
        f.write("adjusted rand score (mean +- std):\n")
        f.write(f"    {np.mean(losses)} +- {np.std(losses)}\n")
        f.write("adjusted rand score per iteration:\n")
        for i, loss in enumerate(losses):
            f.write(f"    Iter {i+1}: {loss}\n")

    print("Results written to .txt file'")








def main():
    parser = argparse.ArgumentParser(description="Evaluate gene cohort stability.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["cohex", "kmeans_by_feature"],
        help="Clustering method to use."
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=5,
        help="Number of iterations to run (default: 5)."
    )
    parser.add_argument(
        "--n_cohorts",
        type=int,
        default=12,
        help="Number of cohorts to generate (default: 12). Specify the optimal no. of cohorts"
    )

    args = parser.parse_args()

    print(f"Running stability_cohort with method={args.method}, n_iter={args.n_iter}, n_cohorts={args.n_cohorts}")
    stability_cohort(method=args.method, n_iter=args.n_iter, n_cohorts=args.n_cohorts)


if __name__ == "__main__":
    main()
