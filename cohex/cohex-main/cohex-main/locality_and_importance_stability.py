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
#----------------Script to find locality and importance stability using optimal k----------------------#


def locality_and_importance_stability(method, n_iter=5, n_cohorts=12):
    """
    Compute and record the locality and importance stability of gene cohorts.

    This function generates cohorts of genes using the specified clustering method
    ('cohex' or 'kmeans_by_feature') and computes:
        1. Locality: how well the cohort explanations are localised.
        2. Importance stability: how stable the feature importance is across iterations.

    The function repeats the process `n_iter` times and saves the results to 
    'output/gene_shap/gene_analysis_results.txt'. All iterations' losses and 
    summary statistics (mean ± std) are recorded.

    Parameters
    ----------
    method : str
        The clustering method to use for generating cohorts. Must be either
        'cohex' or 'kmeans_by_feature'.
    n_iter : int, optional, default=5
        The number of iterations to repeat cohort generation and to 
        estimate locality and importance stability.
    n_cohorts : int, optional, default=12
        The number of cohorts to generate in each iteration.

    Outputs
    -------
    Writes results to two separate (.txt) files of formats: ***_shap_importance_stability.txt
    and ***_shap_locality.txt in output/gene_shap folder
    """   
   
    print("No of total iterations: ",n_iter)
    scenario = GeneScenario()
    save_dir='output/gene_shap'
    os.makedirs(save_dir, exist_ok=True) 

    explainer = SHAPExplainer(scenario.model.predict, mode='default')
 
    locality_losses = np.zeros(n_iter)
    importance_stability_losses = np.zeros(n_iter)

    for i in trange(n_iter):
        # generate cohorts and explanations
        if method == 'kmeans_by_feature':
            k, labels, importance, centroids = kmeans_by_feature(explainer, scenario.X, n_cohorts=n_cohorts) 
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=n_cohorts,
                                          n_iter=5, termination_count=5, verbose=False)
        else:
            raise ValueError(f"Unknown method: {method}")

        # compute both locality and importance stability
        print("Finding locality")
        locality_losses[i] = locality(SHAPExplainer, scenario.model, scenario.X, labels, importance,
                                     classes=[0], mode='default', n_iter=n_iter)
        print("finding stability importance")
        importance_stability_losses[i] = stability_importance(explainer, scenario.X, labels, importance, n_iter=n_iter)

    # write results to file

    with open(os.path.join(save_dir, f'{method}_shap_locality.txt'), 'w') as f:
        f.write("Locality (mean +- std):\n")
        f.write(f"    {np.mean(locality_losses)} +- {np.std(locality_losses)}\n")
        f.write("Locality losses per iteration:\n")
        for i, loss in enumerate(locality_losses):
            f.write(f"    Iter {i+1}: {loss}\n")

    with open(os.path.join(save_dir, f'{method}_shap_importance_stability.txt'), 'w') as f:
        f.write("Importance Stability (mean +- std):\n")
        f.write(f"    {np.mean(importance_stability_losses)} +- {np.std(importance_stability_losses)}\n")
        f.write("Importance Stability losses per iteration:\n")
        for i, loss in enumerate(importance_stability_losses):
            f.write(f"    Iter {i+1}: {loss}\n")
    print("Results written to files'")





def main():
    parser = argparse.ArgumentParser(description="Compute gene locality and importance stability.")
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
        help="Number of iterations for stability computation (default: 5)."
    )
    parser.add_argument(
        "--n_cohorts",
        type=int,
        default=12,
        help="Number of cohorts to generate (default: 12). Specifiy the optimal no. of cohorts"
    )
    args = parser.parse_args()

    print(f"Running locality_and_importance_stability with method={args.method}, n_iter={args.n_iter}, n_cohorts={args.n_cohorts}")
    locality_and_importance_stability(method=args.method, n_iter=args.n_iter, n_cohorts=args.n_cohorts)


if __name__ == "__main__":
    main()