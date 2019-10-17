import copy
import threading
import warnings

import numpy as np
import scipy.stats as stats

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted


def _generate_subsample_indices(random_state, fixed_sample_index, n_samples,
                                n_samples_bootstrap):
    """Utility function to generate a subsample that is constrained to
    contain ``fixed_sample_index``.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap - 1)

    return np.hstack((sample_indices, fixed_sample_index))


def _accumulate_predictions_and_var(predict, X, out, lock):
    """Utility function for joblib's Parallel to accumulate predictions.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[0] += prediction
        out[1] += prediction ** 2


def _forest_predict_var(forest, X_test, n_jobs):
    """Helper function to accumulate predictions and their variances.

    Parameters
    ----------
    forest : RandomForestRegressor
        Regressor object.

    X_test : ndarray, shape (n_test_samples,)
        The design matrix for testing data.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. ``None`` means 1. ``-1`` means
        use all processors.
    """
    check_is_fitted(forest)
    X_test = forest._validate_X_predict(X_test)

    n_jobs, _, _ = _partition_estimators(forest.n_estimators, n_jobs)

    y_hat = np.zeros((X_test.shape[0]), dtype=np.float64)
    y_var = np.zeros((X_test.shape[0]), dtype=np.float64)

    # Parallel loop
    lock = threading.Lock()
    Parallel(n_jobs=n_jobs, verbose=forest.verbose,
             **_joblib_parallel_args(require='sharedmem'))(
        delayed(_accumulate_predictions_and_var)(e.predict, X_test,
                                                 [y_hat, y_var], lock)
        for e in forest.estimators_)

    y_hat /= len(forest.estimators_)
    y_var /= len(forest.estimators_)
    y_var -= y_hat ** 2

    return [y_hat, y_var]


def _shared_fit_and_pred(tree, X_train, y_train, X_test,
                         fixed_sample_index, n_samples_bootstrap,
                         out, lock):
    """Helper function that fits a tree on a subsample that contains
    the sample indexed by ``fixed_sample_index``. In addition, it
    accumalates the predictions in out.

    Parameters
    ----------
    forest : RandomForestRegressor
        Regressor object.

    X_train : ndarray, shape (n_train_samples, n_features)
        The design matrix for training data.

    y_train : ndarray, shape (n_train_samples,)
        The target values for training data.

    X_test : ndarray, shape (n_test_samples,)
        The design matrix for testing data.

    fixed_sample_index : int
        Index of the sample contrained to lie in the bootstrap.

    n_samples_bootstrap : int
        Number of samples in the bootstrap.

    out : list[ndarray]
        List containing the sum of tree predictions.

    lock : threading.Lock
        Lock used to limit out[0] access.
    """
    n_train_samples = X_train.shape[0]

    sample_indices = _generate_subsample_indices(tree.random_state,
                                                 fixed_sample_index,
                                                 n_train_samples,
                                                 n_samples_bootstrap)

    tree.fit(X_train[sample_indices], y_train[sample_indices])

    with lock:
        out[0] += tree.predict(X_test)


def _calc_shared_var(forest, X_train, y_train, X_test,
                     n_fixed_points=50, n_mc_samples=500,
                     n_jobs=1, random_state=None):
    """Calculates the variance of predictions when subsamples share at least
    one sample in common.

    Parameters
    ----------
    forest : RandomForestRegressor
        Regressor object.

    X_train : ndarray, shape (n_train_samples, n_features)
        The design matrix for training data.

    y_train : ndarray, shape (n_train_samples,)
        The target values for training data.

    X_test : ndarray, shape (n_test_samples,)
        The design matrix for testing data.

    n_fixed_points : int, optional (default=50)
        Number of fixed points used to estimate the variance of tree
        predictions with at least one sample in common.

    n_mc_samples : int, optional (default=500)
        Number of monte carlo samples used to estimate the variance of tree
        predictions with at least one sample in common.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. ``None`` means 1. ``-1`` means
        use all processors.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    random_state = check_random_state(random_state)
    n_test_samples = X_test.shape[0]
    n_train_samples = X_train.shape[0]

    y_hat = np.zeros((n_fixed_points, n_test_samples), dtype=np.float64)
    for i in range(n_fixed_points):
        fixed_sample_index = random_state.randint(0, n_train_samples, 1)

        trees = [forest._make_estimator(append=False,
                                        random_state=random_state)
                 for i in range(n_mc_samples)]

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=forest.verbose,
                 **_joblib_parallel_args(prefer='threads',
                                         require='sharedmem'))(
                 delayed(_shared_fit_and_pred)(t, X_train, y_train, X_test,
                                               fixed_sample_index,
                                               forest.max_samples,
                                               [y_hat[i]], lock)
                 for t in trees)

        y_hat[i] /= n_mc_samples

    return np.var(y_hat, axis=0)


def _calc_subsample_var(forest, X_train, y_train, X_test, n_trees_var=500,
                        use_built_trees=True, n_jobs=None, random_state=None):
    """Calculates the variance of predictions due to subsampling.

    Parameters
    ----------
    forest : RandomForestRegressor
        Regressor object.

    X_train : ndarray, shape (n_train_samples, n_features)
        The design matrix for training data.

    y_train : ndarray, shape (n_train_samples,)
        The target values for training data.

    X_test : ndarray, shape (n_test_samples,)
        The design matrix for testing data.

    n_trees_var : int, optional (default=500)
        Number of new trees trained to estimate the variance of tree
        predictions due to subsampling.
        Only used if ``use_internal_variance=True``.

    use_built_trees : bool, optional (default=True)
        Whether to use the already built trees in the forest or build new
        trees when estimating the variance of tree predictions due to
        subsampling..

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. ``None`` means 1. ``-1`` means
        use all processors.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    if use_built_trees:
        y_pred, subsample_var = _forest_predict_var(forest, X_test,
                                                    n_jobs=n_jobs)
    else:
        new_forest = clone(forest)

        param_dict = {'n_estimators': n_trees_var,
                      'n_jobs': n_jobs,
                      'random_state': random_state}
        new_forest.set_params(**param_dict)
        new_forest.fit(X_train, y_train)

        y_pred, subsample_var = _forest_predict_var(new_forest, X_test,
                                                    n_jobs=n_jobs)

    return y_pred, subsample_var


def random_forest_error(forest, X_train, y_train, X_test,
                        alpha_level=0.05,
                        n_fixed_points=50, n_mc_samples=500,
                        n_trees_var=500,
                        use_built_trees=True,
                        random_state=None, n_jobs=None):
    """Calculate error bars from scikit-learn RandomForest regressors.

    Parameters
    ----------
    forest : RandomForestRegressor
        Regressor object.

    X_train : ndarray, shape (n_train_samples, n_features)
        The design matrix for training data.

    y_train : ndarray, shape (n_train_samples,)
        The target values for training data.

    X_test : ndarray, shape (n_test_samples,)
        The design matrix for testing data.

    alpha_level : float, optional (default=0.05)
        The level of the confidence intervals. The error bars are
        (1 - alpha_level) * 100 %  confidence intervals.

    n_fixed_points : int, optional (default=50)
        Number of fixed points used to estimate the variance of tree
        predictions with at least one sample in common.

    n_mc_samples : int, optional (default=500)
        Number of monte carlo samples used to estimate the variance of tree
        predictions with at least one sample in common.

    n_trees_var : int, optional (default=500)
        Number of new trees trained to estimate the variance of tree
        predictions due to subsampling.
        Only used if ``use_internal_variance=True``.

    use_built_trees : bool, optional (default=True)
        Whether to use the already built trees in the forest or build new
        trees when estimating the variance of tree predictions due to
        subsampling..

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel. ``None`` means 1. ``-1`` means
        use all processors.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    y_pred : ndarray, shape (n_test_samples,)
        The prediction at the each point in ``X_test``

    y_err : ndarray, shape (n_test_samples,)
        Asymptotic sample variance for each point in ``X_test``.
        Confidence intervals can be formed as [y_pred - y_err, y_pred + y_err].

    Notes
    -----
    The confidence interval calculation is based on the U-Statistic
    approach described in [Mentch2016]_.

    .. [Mentch2016] L. Mentch and G. Hooker. "Quantifying Uncertainty in
       "Random Forests via Confidence Intervals and Hypothesis Tests",
       Journal of Machine Learning Research vol. 17, pp. 1-41, 2016.
    """
    random_state = check_random_state(random_state)

    # convert a single test point to a numpy array
    if isinstance(X_test, (np.int, np.float)):
        X_test = np.asarray(X_test).reshape(-1, 1)

    # This method only works for random forests grown with subsampling.
    if not hasattr(forest, 'max_samples'):
        raise ValueError("Forests need to support subsampling.")

    # Warn if the number of subsamples is too large for valid inference.
    if forest.max_samples > np.sqrt(X_train.shape[0]):
        warnings.warn("Number of subsamples should be order sqrt(n) for "
                      "valid inference.")

    # calculate predictions as well as between tree variances
    y_pred, subsample_var = _calc_subsample_var(forest,
                                                X_train, y_train, X_test,
                                                n_trees_var=n_trees_var,
                                                use_built_trees=use_built_trees,
                                                n_jobs=n_jobs,
                                                random_state=random_state)

    # calculate variance between subsamples with at least one shared sample
    shared_sample_var = _calc_shared_var(forest, X_train, y_train, X_test,
                                          n_fixed_points=n_fixed_points,
                                          n_mc_samples=n_mc_samples,
                                          random_state=random_state,
                                          n_jobs=n_jobs)

    # 1 - alpha / 2 normal quantile
    z_alpha = stats.norm.ppf(1 - (alpha_level / 2.))

    # estimate training samples to n_trees  ratio
    alpha_hat = X_train.shape[0] / forest.n_estimators

    # asymptotic variance of the normal limiting distribution
    y_err = np.sqrt(
        ((forest.max_samples ** 2) / alpha_hat) * shared_sample_var +
            subsample_var)
    y_err /= np.sqrt(forest.n_estimators)

    # adjustment for level of the test
    y_err *= z_alpha

    return y_pred, y_err
