import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from hypoforest import random_forest_error

n_samples = 200

rng = np.random.RandomState(123)

X = rng.uniform(0, 20, n_samples).reshape(-1, 1)

y = (2 * X + np.sqrt(10) * rng.randn(n_samples, 1)).ravel()

forest = RandomForestRegressor(n_estimators=200,
                               max_samples=30).fit(X, y)

y_pred, y_err = random_forest_error(forest, X, y, X_test=X,
                                    n_fixed_points=50, n_mc_samples=500,
                                    use_built_trees=False, n_jobs=-1,
                                    random_state=42)

plt.errorbar(X.ravel(), y_pred, yerr=y_err, fmt='o',
             markersize=2, ecolor='k', mec='k', mfc='k',
             lw=0.5, capsize=1.0)
plt.plot([0, 20], [0, 40], '--', color='steelblue')
plt.show()
