# CohEx: A Generalized Framework for Cohort Explanation
# Link to the original repo: https://github.com/fy-meng/cohex

Cohort explanations is a type of machine learning explainability that is between global (explaining the whole model) and local (explaining an instance) explanations. It aims to provide information based on subspaces or subsets of samples.

In this repo, the authors implemented _post-hoc_ cohort explanations in `cohort_explanation.py`:

- CohEx: iterative supervised-clustering-based explanation;
- "kmeans-on-feature": naive method that directly applies k-means on the features (ignoring importance);

The following additional metrics have also been implemented by the authors in additional to the clustering loss in `eval.py`:

- Locality: how much change will changes _outside_ a cohort would induce on the explanation on a specific cohort;
- Importance stability: change in explanation if a new sample is added to the dataset.

## Prerequisites to run this code

- Python 3.10;
- `pip install -r requirements.txt`.

## Usage

- `cohort_explanation.cohex` can be used as follows:
    - Parameters:
      - `explainer`: An `Explainer` instance. Should implement the `explain(dataset)` function as defined in `explainer.py`.
      - `dataset`: `np.ndarray`, shape `(n, f)`, where `n` is the number of samples, and `f` is the number of features.
      - `n_cohorts`: _Expected_ number of cohorts.
      - `n_iter`: Number of initializations to run iterative cohort explanation algorithms.
      - `termination_count`: In each iteration, if the number of iteration without improvement exceeds this number, then the iteration would terminate.
      - `verbose`: Whether to print debug information.
      - `return_penalty`: If `True`, then the supervised clustering penalty is also returned. Incompatible with `return_centroids`.
      - `return_centroids`: If `True`, then the centroids are also returned. Incomptible with `return_penalty`.
    - Returns:
      - `k`: `int`, the number of clusters.
      - `labels`: `np.ndarray`, shape `(n,)`, where each entry is an integer in [0, `k`).
      - `cohort_importance`: `np.ndarray`, shape `(k, f)`, the average importance of each cohort.
- kmeans-on-feature can be used in a similar fashion;

