import numpy as np

class SVDPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        
        # SVD directly on centered data — no covariance matrix computed
        # full_matrices=False gives the "thin" SVD: U is (n×k), S is (k,), Vt is (k×p)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        
        k = self.n_components or S.shape[0]
        self.components_ = Vt[:k]               # shape (k, p)
        self.singular_values_ = S[:k]
        self.explained_variance_ = (S[:k] ** 2) / (X.shape[0] - 1)
        total_var = (S ** 2).sum() / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self._U = U[:, :k]
        self._S = S[:k]
        return self

    def transform(self, X):
        Xc = X - self.mean_
        # Efficient: scores = Xc @ Vt.T  OR  U * S for training data
        return Xc @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        # On training data: scores = U * S  (avoids second matrix multiply)
        return self._U * self._S
