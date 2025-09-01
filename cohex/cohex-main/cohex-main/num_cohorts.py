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
#------------------------------Script to find Optimal No. of Cohorts-----------------------------------#



import os
import numpy as np
import matplotlib.pyplot as plt

def plot_penalties(k_values, penalties, method):
    means = np.mean(penalties, axis=1)
    stds = np.std(penalties, axis=1)
    
    # create output directory 
    output_dir = 'output/gene_shap'
    os.makedirs(output_dir, exist_ok=True)
    
    # plot with error bars
    plt.figure(figsize=(8, 5))
    plt.errorbar(k_values, means, yerr=stds, fmt='-o', capsize=5)
    plt.title(f'Penalty vs Number of Cohorts\nMethod: {method}, Explainer: shap')
    plt.xlabel('Number of Cohorts (k)')
    plt.ylabel('Mean Penalty ± Std Dev')
    plt.grid(True)
    
    plot_path = f'{output_dir}/{method}_num_cohorts_shap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  
    print(f"Saved plot to {plot_path}")




def num_cohorts(method):
    """
    Runs the number-of-cohorts experiment.

    This function evaluates clustering performance for different numbers of cohorts (k-values)
    using either the 'cohex' method or the 'kmeans_by_feature' method. It computes penalties
    for each k, prints and saves the results to disk, along with generating a plot.

    Parameters
    ----------
    method : str
        Clustering method to use. Options:
        - 'cohex': Uses the COHEX clustering method.
        - 'kmeans_by_feature': Uses K-means clustering.
        This value is passed from the command line using the `--method` argument.

    Saves
    -----
    - A NumPy `.npy` file containing penalties for each k and trial
      in `output/gene_shap/{method}_num_cohorts_shap.npy`.
    - A plot visualising penalties vs. k.

    Notes
    -----
    The function:
    1. Initialises a `GeneScenario` and a SHAP explainer.
    2. Computes base feature importances.
    3. Runs clustering for k-values [4, 8, 12, 16].
    4. Calculates and stores penalties for each configuration.
    5. Saves results and a plot for analysis.
    """

    scenario = GeneScenario()
    explainer = SHAPExplainer(scenario.model.predict, mode='default')


    base_importance = explainer.explain(scenario.X)

    k_values = [4, 8, 12, 16]
    penalties = np.zeros((len(k_values), 2))  

    for i, k in enumerate(k_values):
        print("i: ",i," k: ",k)
        clustering = SRIDHCR(n_clusters=k)
        for t in range(2):  
            print("trial no: ",t)
            if method == 'kmeans_by_feature':
                _, labels, _, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=k)
                penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
            elif method == 'cohex':
                _, _, _, penalty = cohex(explainer, scenario.X, k, 5, 7, verbose=False, return_penalty=True)
            else:
                raise ValueError

            penalties[i, t] = penalty

    print('penalties:')
    for i, k in enumerate(k_values):
        print(f'  k={k}: {np.mean(penalties[i])} +- {np.std(penalties[i])}')

    output_dir = 'output/gene_shap'   
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/{method}_num_cohorts_shap.npy', penalties)

    # call the visualisation function
    plot_penalties(k_values, penalties, method)


def main():
    parser = argparse.ArgumentParser(description="Run num_cohorts with a given method.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["cohex", "kmeans_by_feature"],
        help="Clustering method to use."
    )
    args = parser.parse_args()

    print(f"running: num_cohorts for shap using {args.method}")
    num_cohorts(method=args.method)




if __name__ == "__main__":
    main()  