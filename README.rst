.. -*- mode: rst -*-

|License|_

.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
.. _License: https://opensource.org/licenses/MIT


.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

Hypothesis Tests for Random Forests
=============================
Currently contains an implementation of the random forest confidence intervals from [Mentch2016]_. This package only implements the external estimation method and is by no means computationally efficient.

Example
-------
.. code-block:: python
    import numpy as np

    from sklearn.ensemble import RandomForestRegressor

    from hypoforest import random_forest_error

    rng = np.random.RandomState(123)

    # generate data
    n_samples = 200
    X = rng.uniform(0, 20, n_samples).reshape(-1, 1)
    y = (2 * X + np.sqrt(10) * rng.randn(n_samples, 1)).ravel()

    # fit a random forest
    forest = RandomForestRegressor(n_estimators=200,
                                   max_samples=30).fit(X, y)

    # calculate confidence intervals for training points
    y_pred, y_err = random_forest_error(forest, X, y, X_test=X)


Dependencies
------------
Hypothesis Tests for Random Forests requires:

- Python (>= 3.6)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Scikit-learn (development)

This packages requires a RandomForestRegressor that supports subsampling, e.g. `max_samples`.

References:
-----------
.. [Mentch2016] L. Mentch and G. Hooker. "Quantifying Uncertainty in
   "Random Forests via Confidence Intervals and Hypothesis Tests",
   Journal of Machine Learning Research vol. 17, pp. 1-41, 2016.
