import pickle
import os

import numpy as np
from sklearn.mixture import GaussianMixture

from ConditionalGMM.condGMM import CondGMM

__all__ = ["GMM"]


class GMM:
    def __init__(self, train_table=None, test_table=None, n_components=13, evaluate_test=True, model_path=None):
        self._train_table = None
        self._test_table = None
        self._train_data = None
        self._test_data = None

        self._n_components = n_components
        self._evaluate_test = evaluate_test

        self._gmm = None
        self._train_score = None
        self._test_score = None
        self._weights = None
        self._means = None
        self._covariances = None
        self._aic = None
        self._bic = None

        if model_path and os.path.exists(model_path) and train_table is None and test_table is None:
            print(f"Loading model from {model_path}")
            self.load_model(model_path)
            return

        if train_table is None or test_table is None:
            raise ValueError("To train a new GMM, you must provide train_table and test_table.")

        self._train_table = train_table[["dmag_r", "dmag_g", "true_mag_r", "true_mag_g"]]
        self._train_data = self._train_table.values
        self._test_table = test_table[["dmag_r", "dmag_g", "true_mag_r", "true_mag_g"]]
        self._test_data = self._test_table.values

        self._train_gmm(model_path=model_path)

    def _train_gmm(self, model_path=None):
        """Train a Gaussian Mixture Model on the training data."""
        self._gmm = GaussianMixture(n_components=self._n_components, random_state=42)
        self._gmm.fit(self._train_data)

        self._train_score = self._gmm.score(self._train_data)
        self._weights = self._gmm.weights_
        self._means = self._gmm.means_
        self._covariances = self._gmm.covariances_
        self._aic = self._gmm.aic(self._train_data)
        self._bic = self._gmm.bic(self._train_data)
        self.conditional_gmm()

        if self._evaluate_test:
            self._test_score = self._gmm.score(self._test_data)

        if model_path:
            self.save_model(model_path)

    def conditional_gmm(self):
        """Build conditional GMMs for sampling."""
        self._cGMM_r = CondGMM(self._weights, self._means, self._covariances, [2])
        self._cGMM_g = CondGMM(self._weights, self._means, self._covariances, [3])
        self.joint_cGMM = CondGMM(self._weights, self._means, self._covariances, [2, 3])

    def save_model(self, model_path=None):
        if model_path is None:
            raise ValueError("Must provide a path to save the model")
        """Save the GMM model using pickle."""
        data = {
            "gmm": self._gmm,
            "weights": self._weights,
            "means": self._means,
            "covariances": self._covariances,
            "n_components": self._n_components,
            "aic": self._aic,
            "bic": self._bic,
            "train_score": self._train_score,
            "test_score": self._test_score,
        }
        with open(model_path, "wb") as f:
            pickle.dump(data, f)
        print(f"GMM model saved to {model_path}")

    def load_model(self, model_path=None):
        if model_path is None:
            raise ValueError("Must provide a path to load the model")

        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self._gmm = data["gmm"]
        self._weights = data["weights"]
        self._means = data["means"]
        self._covariances = data["covariances"]
        self._n_components = data["n_components"]
        self._aic = data["aic"]
        self._bic = data["bic"]
        self._train_score = data["train_score"]
        self._test_score = data["test_score"]
        self.conditional_gmm()
        print(f"GMM model successfully loaded from {model_path}")

    def predict(self, data):
        return self._gmm.predict(data)

    def sample_all(self, n_samples=1):
        samples, _ = self._gmm.sample(n_samples)
        return samples

    def sample_conditional_r(self, true_mag_r, n_samples=1):
        samples = self._cGMM_r.rvs(true_mag_r, size=n_samples)
        return samples.squeeze()

    def sample_conditional_g(self, true_mag_g, n_samples=1):
        samples = self._cGMM_g.rvs(true_mag_g, size=n_samples)
        return samples.squeeze()

    def sample_conditional_joint(self, true_mag_r, true_mag_g, n_samples=1):
        cond_input = np.array([true_mag_r, true_mag_g]).T
        samples = self.joint_cGMM.rvs(cond_input, size=n_samples)
        return samples.squeeze()

    @property
    def train_score(self):
        return self._train_score

    @property
    def test_score(self):
        return self._test_score

    @property
    def weights(self):
        return self._weights

    @property
    def model(self):
        return self._gmm

    @property
    def aic(self):
        return self._aic

    @property
    def bic(self):
        return self._bic
