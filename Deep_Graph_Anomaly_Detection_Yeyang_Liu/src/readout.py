"""CWoLa tree-ensemble readout with the Kitchen Sink classifier recipe.

Each member is a HistGradientBoostingClassifier trained to separate SR (1) from SB (0) events,
with balanced class weights and internal 50/50 early-stopping validation. The anomaly score is the
mean member probability. Member m of run r uses random_state 1 + Nsig + 50 r + m.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


KS_CLASSIFIER = dict(
    loss='log_loss', learning_rate=.1, max_iter=200, max_leaf_nodes=31, max_depth=None,
    min_samples_leaf=20, l2_regularization=0., max_features=1., max_bins=255,
    categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False,
    early_stopping='auto', scoring='loss', validation_fraction=.5, n_iter_no_change=10, tol=1e-7,
    class_weight=None, verbose=0,
)
MEMBERS = 50
_SHARED = {}


def member_seed(run, member, n_signal=1000):
    return 1 + n_signal + MEMBERS*run + member


def class_weights(labels):
    """Weight SR events so both classes carry equal total weight."""
    return np.where(labels == 1, len(labels)/labels.sum() - 1, 1.)


def fit_member(features, labels, *, run=0, member=0, n_signal=1000, threads=2):
    with threadpool_limits(limits=threads):
        model = HistGradientBoostingClassifier(**KS_CLASSIFIER, random_state=member_seed(run, member, n_signal))
        return model.fit(features, labels, sample_weight=class_weights(labels))


def _initialize(features, labels, test, model_dir):
    _SHARED.update(features=features, labels=labels, test=test, model_dir=model_dir)


def _member_scores(item):
    run, member, n_signal = item
    model = fit_member(_SHARED['features'], _SHARED['labels'], run=run, member=member, n_signal=n_signal)
    if _SHARED['model_dir'] is not None:
        joblib.dump(model, Path(_SHARED['model_dir'])/f'member{member:02}.joblib')
    with threadpool_limits(limits=2):
        return model.predict_proba(_SHARED['test'])[:, 1]


def ensemble_scores(features, labels, test, *, run=0, n_signal=1000, workers=6, model_dir=None):
    """Train the 50 members on (features, SR/SB labels); return their mean probability on test.

    With model_dir, every fitted member is saved there for later inference (see predict).
    """
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_initialize,
                             initargs=(features, labels, test, model_dir)) as executor:
        total = 0.
        for scores in executor.map(_member_scores, [(run, member, n_signal) for member in range(MEMBERS)]):
            total = total + scores/MEMBERS
    return total


def predict(model_dir, features, *, threads=8):
    """Mean probability of the saved members on new events, without retraining."""
    total = 0.
    with threadpool_limits(limits=threads):
        for member in range(MEMBERS):
            model = joblib.load(Path(model_dir)/f'member{member:02}.joblib')
            total = total + model.predict_proba(features)[:, 1]/MEMBERS
    return total
