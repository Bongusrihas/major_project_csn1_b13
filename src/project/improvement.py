"""One proposed method: history-aware residual boosting of the paper network.

The neural model stays unchanged. A nonlinear correction learns from current
measurements, past-only history and the network's temporal representation.
All correction trees fit training patients only; validation chooses stopping and
correction strength. No guarantee of a gain is assumed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, mean_absolute_error, average_precision_score
from xgboost import XGBClassifier, XGBRegressor


def history_features(sd, representation):
    x, observed = sd.X.astype(np.float32), sd.M.astype(np.float32)
    count = np.cumsum(observed, axis=1).clip(1)
    mean = np.cumsum(x * observed, axis=1) / count
    previous = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    delta = x - previous
    # Forward-only exponentially weighted history; never inspect a future visit.
    recent = np.zeros_like(x)
    recent[:, 0] = x[:, 0]
    for t in range(1, sd.T):
        recent[:, t] = np.where(observed[:, t] > 0, .5 * x[:, t] + .5 * recent[:, t - 1], recent[:, t - 1])
    # Irregular-visit histories: retain the previous observed measurement rather
    # than interpreting an imputed zero as a real previous value.
    last = np.zeros_like(x[:, 0])
    elapsed = np.zeros_like(last)
    seen = np.zeros_like(last, dtype=bool)
    observed_change, change_rate, observation_age = [], [], []
    for t in range(sd.T):
        elapsed = elapsed + np.expm1(np.clip(sd.gap[:, t], 0, 5))[:, None]
        present = observed[:, t] > 0
        change = np.where(present & seen, x[:, t] - last, 0.)
        observed_change.append(change)
        change_rate.append(change / np.maximum(elapsed, 1.))
        elapsed = np.where(present, 0., elapsed)
        observation_age.append(elapsed.copy())
        last = np.where(present, x[:, t], last)
        seen |= present
    total = np.pad(np.cumsum(x * observed, axis=1), ((0, 0), (1, 0), (0, 0)))
    counts = np.pad(np.cumsum(observed, axis=1), ((0, 0), (1, 0), (0, 0)))
    ends = np.arange(1, sd.T + 1)
    starts = np.maximum(0, ends - 3)
    short_mean = (total[:, ends] - total[:, starts]) / np.maximum(counts[:, ends] - counts[:, starts], 1.)
    pieces = [x, observed, mean, recent, delta, short_mean,
              np.stack(observed_change, axis=1), np.stack(change_rate, axis=1),
              np.log1p(np.stack(observation_age, axis=1)), sd.gap[..., None], representation]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def auc(y, p):
    return float(roc_auc_score(y, p)) if np.unique(y).size == 2 else .5


def patient_weights(sd):
    """Give each training patient equal total influence despite visit counts."""
    counts = sd.valid.sum(axis=1).clip(1)
    weights = np.broadcast_to((1. / counts)[:, None], sd.valid.shape)[sd.valid]
    return (weights / weights.mean()).astype(np.float32)


def ranking_score(y, probability):
    # Both tie breakers are evaluated on validation patients only.
    pr = float(average_precision_score(y, probability)) if np.any(y) else 0.
    return auc(y, probability), pr, -float(np.mean((y - probability) ** 2))


@dataclass
class TemporalResidualBoosting:
    classifier: object = None
    regressor: object = None
    flare_strength: float = 0.
    severity_strength: float = 0.
    validation: dict = None

    def fit(self, train, val, train_base, val_base, train_h, val_h, seed=42):
        xt = history_features(train, train_h)[train.valid]
        xv = history_features(val, val_h)[val.valid]
        yt, yv = train.y_flare[train.valid], val.y_flare[val.valid]
        mt = logit(train_base['p'][train.valid].clip(1e-5, 1 - 1e-5))
        mv = logit(val_base['p'][val.valid].clip(1e-5, 1 - 1e-5))
        shared = dict(n_estimators=300, learning_rate=.03, subsample=.85,
                      colsample_bytree=.8, tree_method='hist', n_jobs=4,
                      random_state=seed, early_stopping_rounds=25)
        # Fixed, small search inside the one proposed framework. No test data is
        # accepted by this interface, and no head is chosen using test scores.
        configurations = [
            dict(max_depth=2, min_child_weight=12, reg_lambda=20., balanced=False),
            dict(max_depth=3, min_child_weight=8, reg_lambda=10., balanced=False),
            dict(max_depth=4, min_child_weight=8, reg_lambda=15., balanced=True),
        ]
        best_rank = ranking_score(yv, val_base['p'][val.valid])
        best_mae = float(mean_absolute_error(val.y_sev[val.valid], val_base['sev'][val.valid].clip(0, 4)))
        self.classifier, self.regressor = None, None
        self.flare_strength, self.severity_strength = 0., 0.
        search = []
        selected_flare, selected_severity = None, None
        residual = train.y_sev[train.valid] - train_base['sev'][train.valid]
        val_residual = val.y_sev[val.valid] - val_base['sev'][val.valid]
        for index, configuration in enumerate(configurations):
            params = {key: value for key, value in configuration.items() if key != 'balanced'}
            weights = patient_weights(train) if configuration['balanced'] else None
            record = {'configuration': index, **configuration}
            if np.unique(yt).size == 2:
                classifier = XGBClassifier(**shared, **params, eval_metric='logloss')
                classifier.fit(xt, yt, sample_weight=weights, base_margin=mt,
                    eval_set=[(xv, yv)], base_margin_eval_set=[mv], verbose=False)
                corrected = classifier.predict(xv, base_margin=mv, output_margin=True)
                strength = max((0., .25, .5, .75, 1.), key=lambda a: ranking_score(yv, expit(mv + a * (corrected - mv))))
                score = ranking_score(yv, expit(mv + strength * (corrected - mv)))
                record.update(validation_auroc=score[0], validation_auprc=score[1])
                if score > best_rank:
                    self.classifier, self.flare_strength = classifier, strength
                    best_rank, selected_flare = score, index
            regressor = XGBRegressor(**shared, **params, eval_metric='rmse')
            regressor.fit(xt, residual, sample_weight=weights, eval_set=[(xv, val_residual)], verbose=False)
            correction = regressor.predict(xv)
            strength = min((0., .25, .5, .75, 1.), key=lambda a: mean_absolute_error(
                val.y_sev[val.valid], np.clip(val_base['sev'][val.valid] + a * correction, 0, 4)))
            error = float(mean_absolute_error(val.y_sev[val.valid], np.clip(val_base['sev'][val.valid] + strength * correction, 0, 4)))
            record['validation_mae'] = error
            if error < best_mae:
                self.regressor, self.severity_strength = regressor, strength
                best_mae, selected_severity = error, index
            search.append(record)
        self.validation = {'flare_strength': self.flare_strength,
                           'severity_strength': self.severity_strength,
                           'classifier_trees': self.classifier.best_iteration + 1 if self.classifier else 0,
                           'severity_trees': self.regressor.best_iteration + 1 if self.regressor else 0,
                           'selected_flare_configuration': selected_flare,
                           'selected_severity_configuration': selected_severity,
                           'configuration_search': search}
        return self

    def predict(self, sd, base, representation):
        shape = sd.X.shape[:2]
        features = history_features(sd, representation)
        features = features.reshape(-1, features.shape[-1])
        margin = logit(base['p'].ravel().clip(1e-5, 1 - 1e-5))
        if self.classifier is not None:
            corrected = self.classifier.predict(features, base_margin=margin, output_margin=True)
            probability = expit(margin + self.flare_strength * (corrected - margin)).reshape(shape)
        else:
            probability = base['p'].copy()
        severity = base['sev'].clip(0, 4)
        if self.regressor is not None:
            severity = np.clip(base['sev'] + self.severity_strength * self.regressor.predict(features).reshape(shape), 0, 4)
        return {'p': probability, 'sev': severity}
