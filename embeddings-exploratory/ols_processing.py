from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from pickle import load, dump, HIGHEST_PROTOCOL
from joblib import Parallel, delayed

N_JOBS = 6
BOOTSTRAP_SIZE = 1000
RANDOM_SEED = 375


def run_ols(X: np.array, Y: np.array) -> Dict[str, np.array]:
    assert X.shape[0] == Y.shape[0]
    # LinearRegression does not impose penalty by default
    full_sample_out = LinearRegression(fit_intercept=False).fit(X, Y)
    beta_coefficients = full_sample_out.coef_
    normed_betas = np.linalg.norm(beta_coefficients, axis=0)
    return {"betas": beta_coefficients, "normed_betas": normed_betas}


def permute_ols(
    token: Dict[str, np.array], rng: np.random.Generator, confidence: float = 90
) -> Dict[str, np.array]:
    def sample():
        Y_permuted = rng.permutation(Y)
        return run_ols(X, Y_permuted)

    X, Y = token["X"], token["Y"]
    assert X.shape[0] == Y.shape[0]
    normed_betas = [sample()["normed_betas"] for _ in range(BOOTSTRAP_SIZE)]

    # Getting confidence interval
    offset = (100 - confidence) / 2
    ci_normed_betas = np.percentile(normed_betas, [offset, confidence + offset], axis=0)

    # Conducting permutation test: p-value is %values "more extreme" than ground_truth
    ground_truth = run_ols(X, Y)["normed_betas"]
    empirical_pvalue = np.apply_along_axis(
        lambda a: sum(1 if x > ground_truth else 0 for x in a) / len(a),
        axis=0,
        arr=normed_betas,
    )

    return {
        "token": token["token"],
        "ground truth": ground_truth,
        "CI": ci_normed_betas,
        "p-value": empirical_pvalue,
    }


if __name__ == "__main__":
    with open("alc_stepone.pickle", "rb") as handle:
        alc_stepone = load(handle)

    ss = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = ss.spawn(len(alc_stepone))

    with Parallel(n_jobs=N_JOBS, verbose=100, prefer="threads") as parallel:
        streams = parallel(delayed(np.random.default_rng)(seed) for seed in child_seeds)
        alc_steptwo = parallel(
            delayed(permute_ols)(token, rng) for token, rng in zip(alc_stepone, streams)
        )

    with open("alc_steptwo.pickle", "wb") as handle:
        dump(alc_steptwo, handle, protocol=HIGHEST_PROTOCOL)
