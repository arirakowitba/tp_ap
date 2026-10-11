"""Frozen selection, alternate-split confirmation and conditional full refit.

The additional media inputs are numeric listing metadata. This workflow has no
image pixels, video frames, external item data, or downloaded visual features.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def aligned_predictions(path, rows, labels=None):
    frame = pd.read_csv(path)
    if 'row_index' not in frame or frame.row_index.duplicated().any():
        raise ValueError(f'Invalid row IDs: {path}')
    if set(frame.row_index) != set(map(int, rows)) or len(frame) != len(rows):
        raise ValueError(f'Evaluation row set mismatch: {path}')
    frame = frame.set_index('row_index').loc[rows]
    label_col = 'y_used' if 'y_used' in frame else 'y'
    if labels is not None and (label_col not in frame or not np.array_equal(frame[label_col], labels)):
        raise ValueError(f'Evaluation labels mismatch: {path}')
    probability_col = 'probability_used' if 'probability_used' in frame else 'p'
    p = frame[probability_col].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f'Invalid probabilities: {path}')
    return p


def score(y, p, threshold):
    prediction = p >= threshold
    return dict(accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, average='macro')),
                logloss=float(log_loss(y, p, labels=[0, 1])), threshold=float(threshold))


def paired(y, prior, candidate, threshold):
    old = (prior >= .5) == y
    new = (candidate >= threshold) == y
    repairs = int(np.sum(~old & new))
    introduces = int(np.sum(old & ~new))
    n = repairs + introduces
    return dict(repairs=repairs, introduces=introduces, net_correct=repairs-introduces,
                discordant=n, exact_two_sided_p=float(binomtest(repairs, n, .5).pvalue) if n else 1.)


def select(args):
    out = Path(args.output)
    frozen_path = out / 'round4_selected.json'
    if frozen_path.exists():
        raise FileExistsError('Selection is already frozen; use a new output directory for a new protocol.')
    evaluation = pd.read_csv(Path(args.cache) / 'evaluation_rows.csv')
    rows = evaluation.row_index.to_numpy()
    y = evaluation.y_used.to_numpy()
    if len(rows) != 21000 or len(np.unique(rows)) != len(rows):
        raise ValueError('Selection requires 21k unique evaluation rows.')
    protocol = json.loads((Path(args.cache) / 'protocol.json').read_text())
    fp = protocol['fingerprint']
    if fp.get('seed') != 42 or fp.get('holdout_folds') != 3:
        raise ValueError('Selection requires seed42, holdout3 (49k/21k).')
    prior = aligned_predictions(args.prior_selection, rows, y)
    predictions = {'prior': prior}
    configs = {'prior': dict(name='prior', family='prior', threshold=.5)}
    candidates = []
    for path in sorted(Path(args.candidates).rglob('r4_*.json')):
        cfg = json.loads(path.read_text())
        name = cfg.get('name')
        if not name or cfg.get('family') not in {'cat', 'lgb'}:
            continue
        prediction_path = path.parent / (name + '_predictions.csv')
        if not prediction_path.exists():
            continue
        # Do not accept incomplete training configurations or visual claims.
        if not cfg.get('params') or any(cfg.get(k, False) for k in
             ('uses_pixels', 'uses_video_frames', 'image_pixels_used', 'video_frames_used')):
            raise ValueError(f'Invalid metadata-only configuration: {name}')
        p = aligned_predictions(prediction_path, rows, y)
        cfg_fp = cfg.get('fingerprint')
        if cfg_fp and cfg_fp != fp:
            raise ValueError(f'Training fingerprint mismatch: {name}')
        predictions[name], configs[name] = p, cfg
        group = 'lgb' if cfg['family'] == 'lgb' else cfg['config']
        # Complementarity with the existing best ensemble is a selection objective,
        # not a confirmation statistic. The 60/40 comparator is fixed in advance.
        rank=max(float(np.mean((p>=.5)==y)),float(np.mean((.6*prior+.4*p>=.5)==y)))
        candidates.append((group, name, rank))
    if not candidates:
        raise ValueError('No completed candidate predictions.')
    # At most one model per architecture, ranked by standalone or fixed 60/40 blend.
    tops = []
    for group in sorted({item[0] for item in candidates}):
        tops.append(sorted([item for item in candidates if item[0] == group], key=lambda x: (-x[2], x[1]))[0][1])
    blend_defs = [{name: 1.} for name in ['prior'] + tops]
    for name in tops:
        blend_defs.extend([{'prior': round(1-w, 1), name: w} for w in (.1, .2, .3, .4, .5, .6, .7, .8, .9)])
    for a, b in itertools.combinations(tops, 2):
        for prior_weight in (.4, .5, .6, .7, .8):
            for ai in range(1, int(round((1-prior_weight)*10))):
                wa = ai/10
                blend_defs.append({'prior': prior_weight, a: wa, b: round(1-prior_weight-wa, 1)})
    if len(tops) == 3:
        by_group = {group: name for group, name, _ in candidates if name in tops}
        blend_defs.append({by_group['lgb']: .6, by_group['enriched_d9']: .2,
                           by_group['native_text_d7']: .2})
    trials = []
    best = None
    best_key = None
    for weights in blend_defs:
        p = sum(predictions[name]*weight for name, weight in weights.items())
        for q in (.49, .5, .51):
            metrics = score(y, p, q)
            trials.append(dict(weights=json.dumps(weights, sort_keys=True), **metrics))
            key = (metrics['accuracy'], -abs(q-.5), -len(weights), weights.get('prior', 0))
            if best_key is None or key > best_key:
                best_key, best = key, dict(weights=weights, threshold=q, selection_metrics=metrics)
    selected_p = sum(predictions[name]*w for name, w in best['weights'].items())
    frozen = dict(version=1, stage='selection_frozen', frozen_at=time.time(),
                  fingerprint=fp, selection_row_sha256=row_hash(rows), selection_rows=len(rows),
                  components={name: configs[name] for name in best['weights']},
                  architectures_considered=tops, trials=len(trials),
                  architecture_ranking='max standalone or fixed .6prior+.4candidate accuracy at .5 on selection only',
                  selection_baseline=score(y, prior, .5), selection_paired=paired(y, prior, selected_p, best['threshold']),
                  image_pixels_used=False, video_frames_used=False, inputs='text_and_listing_media_metadata',
                  public_accuracy=None, **best)
    frozen['selection_digest'] = digest(frozen)
    atomic_json(frozen_path, frozen)
    pd.DataFrame(trials).to_csv(out / 'round4_blend_trials.csv', index=False)
    pd.DataFrame(dict(row_index=rows, y_used=y, probability_used=selected_p)).to_csv(
        out / 'round4_selection_predictions.csv', index=False)
    print(json.dumps(dict(stage='selection', **best, baseline=frozen['selection_baseline'], trials=len(trials))), flush=True)
    return frozen


def load_frozen(path):
    frozen = json.loads(Path(path).read_text())
    stated = frozen.pop('selection_digest')
    if digest(frozen) != stated:
        raise ValueError('Frozen selection digest changed.')
    frozen['selection_digest'] = stated
    return frozen


def add_media(data, tr, va, args, full, columns=None):
    train = pd.read_csv(args.media_train).set_index('row_index', verify_integrity=True)
    other = pd.read_csv(args.media_test).set_index('row_index', verify_integrity=True) if full else train
    columns = columns or [c for c in train if c.startswith('media_')]
    if not columns or any(c in tr for c in columns):
        raise ValueError('Invalid or duplicate media metadata columns.')
    for frame, media, rows in ((tr, train, data['t']), (va, other, data['v'])):
        values = media.loc[rows, columns].reset_index(drop=True)
        for c in columns:
            frame[c] = pd.to_numeric(values[c], errors='raise').to_numpy(dtype='float32')
    return tr, va


def fit_component(cfg, args, full=False):
    cache_name = 'stack.pkl' if cfg['family'] == 'lgb' else 'native.pkl'
    data = pd.read_pickle(Path(args.cache) / cache_name)
    expected_fp = json.loads((Path(args.cache) / 'protocol.json').read_text())['fingerprint']
    if data.get('fingerprint') != expected_fp:
        raise ValueError('Feature cache/protocol fingerprint mismatch.')
    if len(data['tr']) != len(data['y_train']) or len(data['va']) != len(data['v']):
        raise ValueError('Cache feature/row counts mismatch.')
    tr, va = data['tr'].copy(), data['va'].copy()
    cats = list(data['cats'])
    if cfg.get('use_media', cfg.get('media_metadata', False)):
        tr, va = add_media(data, tr, va, args, full, cfg.get('media_columns'))
    params = dict(cfg['params'])
    if cfg['family'] == 'lgb':
        from lightgbm import LGBMClassifier
        for c in cats:
            dtype = pd.CategoricalDtype(categories=pd.Index(tr[c].unique()))
            tr[c], va[c] = tr[c].astype(dtype), va[c].astype(dtype)
        params['n_jobs'] = 2
        model = LGBMClassifier(**params)
        model.fit(tr, data['y_train'])
    elif cfg['family'] == 'cat':
        from catboost import CatBoostClassifier
        texts = list(cfg.get('text_features', []))
        if texts:
            for frame, key in ((tr, 'text_tr'), (va, 'text_va')):
                frame['full_text'] = pd.Series(data[key]).fillna('').astype(str).replace('', 'empty').to_numpy()
        for c in cats:
            tr[c] = tr[c].astype('string').fillna('__MISSING__').astype(str)
            va[c] = va[c].astype('string').fillna('__MISSING__').astype(str)
        numeric = tr.columns.difference(cats + texts)
        tr[numeric] = tr[numeric].replace([np.inf, -np.inf], np.nan)
        va[numeric] = va[numeric].replace([np.inf, -np.inf], np.nan)
        model = CatBoostClassifier(**params, loss_function='Logloss', eval_metric='Accuracy',
                  thread_count=2, allow_writing_files=False, one_hot_max_size=10, random_strength=.5)
        model.fit(tr, data['y_train'], cat_features=cats, text_features=texts, verbose=100)
    else:
        raise ValueError('Unknown model family.')
    p = model.predict_proba(va)[:, 1]
    if len(p) != len(data['v']) or not np.isfinite(p).all():
        raise ValueError('Invalid fitted probabilities.')
    return p


def confirm(args):
    frozen = load_frozen(args.selected)
    out = Path(args.output)
    protocol = json.loads((Path(args.cache) / 'protocol.json').read_text())
    fp = protocol['fingerprint']
    if fp.get('train_sha256') != frozen['fingerprint']['train_sha256'] or fp.get('seed') != 731 or fp.get('holdout_folds') != 2:
        raise ValueError('Confirmation requires same dataset and seed731, holdout2.')
    evaluation = pd.read_csv(Path(args.cache) / 'evaluation_rows.csv')
    rows, y = evaluation.row_index.to_numpy(), evaluation.y_used.to_numpy()
    if len(rows) != 14000 or len(np.unique(rows)) != len(rows):
        raise ValueError('Confirmation requires 14k unique evaluation rows.')
    selection_rows = pd.read_csv(Path(args.selected).parent / 'round4_selection_predictions.csv').row_index.to_numpy()
    if row_hash(selection_rows) != frozen['selection_row_sha256']:
        raise ValueError('Frozen selection row manifest changed.')
    overlap = len(set(map(int, rows)) & set(map(int, selection_rows)))
    prior = aligned_predictions(args.prior_confirmation, rows, y)
    probabilities = {}
    for name, cfg in frozen['components'].items():
        p = prior if name == 'prior' else fit_component(cfg, args)
        probabilities[name] = p
        pd.DataFrame(dict(row_index=rows, y_used=y, probability_used=p)).to_csv(
            out / (name + '_confirmation_predictions.csv'), index=False)
    # Read confirmation metrics only after all frozen component fits finish.
    p = sum(probabilities[name]*w for name, w in frozen['weights'].items())
    candidate, baseline = score(y, p, frozen['threshold']), score(y, prior, .5)
    promote = (frozen['selection_metrics']['accuracy'] > frozen['selection_baseline']['accuracy']
               and candidate['accuracy'] > baseline['accuracy'])
    result = dict(stage='confirmation_completed', selection_digest=frozen['selection_digest'],
        fingerprint=fp, evaluation_rows=len(rows), confirmation_row_sha256=row_hash(rows),
        selection_confirmation_overlap=overlap,
        validation_caveat='Alternate grouped split of the same labeled dataset; selection and confirmation evaluation rows overlap. This is not an untouched independent test set.',
        candidate=candidate, baseline=baseline, paired=paired(y, prior, p, frozen['threshold']),
        promote=bool(promote), local_goal_reached=bool(candidate['accuracy'] >= .95 and frozen['selection_metrics']['accuracy'] >= .95),
        public_accuracy=None, image_pixels_used=False, video_frames_used=False,
        components={name: score(y, prob, .5) for name, prob in probabilities.items()})
    atomic_json(out / 'round4_confirmation.json', result)
    pd.DataFrame(dict(row_index=rows, y_used=y, probability_used=p)).to_csv(out / 'round4_confirmation_predictions.csv', index=False)
    print(json.dumps(result), flush=True)
    return result


def refit(args):
    frozen = load_frozen(args.selected)
    confirmation = json.loads(Path(args.confirmation).read_text())
    if confirmation['selection_digest'] != frozen['selection_digest'] or not confirmation['promote']:
        raise ValueError('Full refit requires a matching improved alternate-split confirmation.')
    protocol = json.loads((Path(args.cache) / 'protocol.json').read_text())
    fp = protocol['fingerprint']
    reference_metadata = json.loads(Path(args.prior_metadata).read_text())
    if (fp.get('train_sha256') != frozen['fingerprint']['train_sha256']
        or fp.get('train_sha256') != reference_metadata['train_sha256']
        or fp.get('test_sha256') != reference_metadata['test_sha256']
        or fp.get('full') is not True):
        raise ValueError('Full refit requires full:true and exact train/test reference SHA256.')
    data = pd.read_pickle(Path(args.cache) / 'native.pkl')
    if data.get('fingerprint') != fp:
        raise ValueError('Full native cache fingerprint differs from protocol.')
    if len(data['t']) != 70000 or len(data['v']) != 30000 or data.get('y_eval') is not None:
        raise ValueError('Full refit requires 70k labeled train and 30k unlabeled test rows.')
    reference = pd.read_csv(args.prior_full)
    ids = np.asarray(data.get('test_ids', np.arange(1, 30001)))
    if reference.ID.duplicated().any() or not np.array_equal(reference.ID, ids):
        raise ValueError('Full reference ID ordering differs.')
    prior = reference.probability_used.to_numpy(dtype=float)
    if not np.isfinite(prior).all() or np.any((prior < 0) | (prior > 1)):
        raise ValueError('Invalid full baseline probabilities.')
    del data
    probabilities = {}
    out = Path(args.output)
    for name, cfg in frozen['components'].items():
        p = prior if name == 'prior' else fit_component(cfg, args, full=True)
        probabilities[name] = p
        pd.DataFrame(dict(ID=ids, probability_used=p)).to_csv(out / (name+'_full_predictions.csv'), index=False)
    p = sum(probabilities[name]*w for name, w in frozen['weights'].items())
    submission = Path(args.submission)
    submission.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ID': ids, 'condition': np.where(p >= frozen['threshold'], 'used', 'new')}).to_csv(submission, index=False)
    pd.DataFrame(dict(ID=ids, probability_used=p)).to_csv(out / 'round4_full_predictions.csv', index=False)
    result = dict(stage='full_refit_completed', selection_digest=frozen['selection_digest'],
        submission=str(submission.resolve()), rows=len(p), threshold=frozen['threshold'],
        submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest(),
        changed_vs_prior=int(np.sum((p >= frozen['threshold']) != (prior >= .5))),
        image_pixels_used=False, video_frames_used=False, public_accuracy=None)
    atomic_json(out / 'round4_refit.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'confirm', 'refit'])
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', default=str(ROOT / 'results/round_v4'))
    parser.add_argument('--candidates', default=str(ROOT / 'results/round_v4'))
    parser.add_argument('--selected', default=str(ROOT / 'results/round_v4/round4_selected.json'))
    parser.add_argument('--confirmation', default=str(ROOT / 'results/round_v4/round4_confirmation.json'))
    parser.add_argument('--media-train', default=str(ROOT / 'results/round_v4/media_features_train.csv'))
    parser.add_argument('--media-test', default=str(ROOT / 'results/round_v4/media_features_test.csv'))
    parser.add_argument('--prior-selection', default=str(ROOT / 'optimization/holdout_predictions.csv'))
    parser.add_argument('--prior-confirmation', default=str(ROOT / 'results/round_v3/prior_confirmation_predictions.csv'))
    parser.add_argument('--prior-metadata', default=str(ROOT / 'results/round_v3/prior_prediction_metadata.json'))
    parser.add_argument('--prior-full', default=str(ROOT / 'results/round_v3/prior_full_predictions.csv'))
    parser.add_argument('--submission', default=str(ROOT / 'submissions/submission_round4_confirmed.csv'))
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    if args.deadline <= time.time():
        raise TimeoutError('Round deadline expired.')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Round deadline reached.')))
    signal.alarm(max(1, int(args.deadline-time.time())))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    {'select': select, 'confirm': confirm, 'refit': refit}[args.stage](args)


if __name__ == '__main__':
    main()
