import numpy as np
import matplotlib.pyplot as plt


def _in_arr(row: np.ndarray, arr: np.ndarray):
    return any((arr[:] == row).all(axis=1))


class AlternateExplainee:
    def __init__(self, model, cohort, classes, p=0.05):
        self.model = model
        self.cohort = cohort
        self.classes = classes
        self.p = p

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        # ensure X is 2D
        X = np.atleast_2d(X)
        
        pred_actual = self.model.predict(X)
        
        # generate random regression outputs with same mean and std as the actual predictions
        mean = np.mean(pred_actual)
        std = np.std(pred_actual)
        pred_rand = np.random.normal(loc=mean, scale=std, size=X.shape[0])
        
        # Mask: true if point is not in cohort AND should be perturbed
        idx = np.array([not _in_arr(x, self.cohort) for x in X]) & (np.random.rand(X.shape[0]) < self.p)
        
        # replace only the selected predictions with random ones
        output = pred_actual.copy()
        output[idx] = pred_rand[idx]
        
        # if input was originally 1D, return scalar
        return output[0] if len(X) == 1 else output




def locality(explainer_class, model, dataset, labels, importance, classes, n_iter=10, **explainer_kwargs):
    k = len(np.unique(labels))

    loss = 0.0
    for j in range(k):
        cohort = dataset[labels == j]
        for t in range(n_iter):
            model_alt = AlternateExplainee(model, cohort, classes)
            explainer = explainer_class(model_alt, **explainer_kwargs)
            importance_alt = np.mean(explainer.explain(cohort), axis=0)
            loss += np.sum((importance_alt - importance[j]) ** 2) / k / n_iter

    return loss




def stability_importance(explainer, dataset, labels, importance, n_iter=10, recompute=False):
    k = len(np.unique(labels))

    importance_all = explainer.explain(dataset)

    loss = 0.0
    for j in range(k):
        cohort = dataset[labels == j]
        non_cohort = dataset[labels != j]
        cohort_indices = np.where(labels == j)[0]
        non_cohort_indices = np.where(labels != j)[0]
        
        for t in range(n_iter):
            sample_idx_in_non_cohort = np.random.choice(non_cohort.shape[0])
            sample_idx = non_cohort_indices[sample_idx_in_non_cohort]  # actual index in dataset
            
            if recompute:
                # combine cohort indices and the single outside sample index
                indices = np.append(cohort_indices, sample_idx)
                importance_alt = np.mean(importance_all[indices], axis=0)
            else:
                sample = non_cohort[sample_idx_in_non_cohort]
                cohort_alt = np.vstack((cohort, sample[np.newaxis]))
                importance_alt = np.mean(explainer.explain(cohort_alt), axis=0)

            loss += np.sum((importance_alt - importance[j]) ** 2) / k / n_iter

    return loss

