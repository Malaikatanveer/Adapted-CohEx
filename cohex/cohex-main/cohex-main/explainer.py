import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import torch
import shap


class Explainer:
    def __init__(self, explainee):
        self.explainee = explainee

    def explain(self, dataset):
        raise NotImplemented


class SHAPExplainer(Explainer):
    def __init__(self, explainee, mode='default', index=None):
        super(SHAPExplainer, self).__init__(explainee)
        assert mode in ('default', 'kernel', 'deep')
        self.mode = mode
        print("inside shap explainer innit")
        # only outputs importance of columns defined by INDEX
        self.index = index

    def explain(self, dataset):
        print("inside shap explainer explain")
        if self.mode == 'default':
            explainer = shap.Explainer(self.explainee, dataset)
            result = explainer(dataset).values
        elif self.mode == 'deep':
            dataset = torch.Tensor(dataset)
            explainer = shap.DeepExplainer(self.explainee, dataset)
            result = explainer.shap_values(dataset)
            result = np.array(result)
            result = result.squeeze(axis=2)
            result = result.transpose((1, 0, 2, 3))
        else:
            raise ValueError

        result = np.array(result)
        if self.index is not None:
            result = result[:, self.index]
        print("resturning Explanation results")
        return result


