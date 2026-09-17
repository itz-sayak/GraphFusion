import numpy as np

from backend.matching.learned import FEATURES, LearnedMatcher, pair_features


def _toy(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, len(FEATURES)))
    X[:, 0] += 2.5 * y  # one informative feature
    return X, y


def test_fit_predict_probabilities():
    X, y = _toy()
    m = LearnedMatcher().fit(X, y, groups=np.arange(len(y)) % 5)
    p = m.predict(X)
    assert p.shape == (len(y),) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3
    assert m.metadata["oof_auc"] > 0.8


def test_retrain_with_decision_moves_that_pair(tmp_path):
    X, y = _toy()
    m = LearnedMatcher().fit(X, y, groups=np.arange(len(y)) % 5)
    m.save(tmp_path / "m.joblib", training_data=(X, y))
    loaded = LearnedMatcher.load(tmp_path / "m.joblib")
    x = np.zeros((1, len(FEATURES)))
    x[0, 0] = 3.0  # looks like a match
    before = loaded.predict(x)[0]
    after = loaded.retrain_with_decisions(np.repeat(x, 3, axis=0), np.zeros(3, dtype=int), weight=50).predict(x)[0]
    assert after < before, "a user rejection lowers the probability of that pair"


def test_features_from_real_evidence(merged_session):
    s = merged_session
    (ka, kb), ev = next(iter(s.matching.evidence.items()))
    f = pair_features(ev, s.matching.columns[ka], s.matching.columns[kb])
    assert len(f) == len(FEATURES) and all(np.isfinite(f))
