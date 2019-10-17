import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_random_state

from hypoforest.confidence_interval import random_forest_error


def make_slr(n_samples=200, random_state=None):
    random_state = check_random_state(random_state)

    X = random_state.uniform(0, 20, n_samples).reshape(-1, 1)
    y = (2 * X + np.sqrt(10) * random_state.randn(n_samples, 1)).ravel()

    return X, y

n_iter = 250
results = np.zeros((n_iter, 2), dtype=np.float64)
for iter_idx in range(n_iter):
    X, y = make_slr(n_samples=200, random_state=iter_idx)

    forest = RandomForestRegressor(n_estimators=200,
                                   max_samples=30, n_jobs=-1).fit(X, y)

    y_pred, y_err = random_forest_error(forest, X, y, X_test=10.,
                                        alpha_level=0.05,
                                        n_fixed_points=50, n_mc_samples=500,
                                        n_estimators_var=500,
                                        use_internal_variance=False,
                                        random_state=iter_idx * 1245,
                                        n_jobs=-1)
    results[iter_idx, 0] = y_pred[0]
    results[iter_idx, 1] = y_err[0]

y_bar = results[:, 0].mean(axis=0)
plt.errorbar(np.arange(n_iter), results[:, 0], yerr=results[:, 1],
             ecolor='k', mec='k', mfc='k')
plt.plot([0, n_iter], [y_bar, y_bar], 'k--')
plt.show()

