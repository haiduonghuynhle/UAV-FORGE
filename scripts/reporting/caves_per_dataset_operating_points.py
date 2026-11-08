#!/usr/bin/env python3
"""Per-dataset CAVES threshold sweep on clean (natural) runs.

For each dataset, sweep tau over caves_risk_score among accepted frames:
accept frame if score <= tau. Report coverage/ABR operating points.
Bad pose = post_fgo error > dataset threshold (20m AVL/VPAIR, 150m UAV-VisLoc).
"""
import csv
import glob
import math

ROOTS = [
    '/workspace/forge/code/results_server/forge_proposed_20260628_032310',
    '/workspace/forge/code/results_server/forge_proposed_uavvisloc_20260628_190949',
]
PIPELINES = ['ours_ultravpr_cascade_6dof_e2e', 'ours_camp_cascade_6dof_e2e']
DATASETS = {'vpair': 20.0, 'avl': 20.0, 'uav_visloc': 150.0}


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def sf(v):
    try:
        x = float(v)
        return x if x == x and abs(x) != float('inf') else None
    except (TypeError, ValueError):
        return None


def load(dataset, thr_m):
    frames = []
    for pipe in PIPELINES:
        paths = []
        for root in ROOTS:
            paths.extend(glob.glob(f'{root}/{dataset}/{pipe}/*/predictions.csv'))
        for path in paths:
            if '/smoke/' in path:
                continue
            with open(path) as f:
                for row in csv.DictReader(f):
                    if row.get('accepted_for_metrics') != 'True':
                        continue
                    score = sf(row.get('caves_risk_score'))
                    if score is None:
                        continue
                    err = sf(row.get('post_fgo_error_m'))
                    if err is None:
                        g_lat, g_lon = sf(row.get('gt_lat')), sf(row.get('gt_lon'))
                        p_lat, p_lon = sf(row.get('pred_lat')), sf(row.get('pred_lon'))
                        if None in (g_lat, g_lon, p_lat, p_lon):
                            continue
                        err = haversine_m(g_lat, g_lon, p_lat, p_lon)
                    frames.append((score, err > thr_m, pipe))
    return frames


def auroc(frames):
    """Rank AUROC (Mann-Whitney U with half credit for ties).

    `frames` are (score, is_bad) pairs; anything after the second element is
    ignored, so this module's own (score, is_bad, pipeline) triples still work.
    """
    pos = sorted(s for s, bad, *_ in frames if bad)
    neg = sorted(s for s, bad, *_ in frames if not bad)
    if not pos or not neg:
        return float('nan')
    import bisect
    wins = 0.0
    for s in neg:
        lo = bisect.bisect_left(pos, s)
        hi = bisect.bisect_right(pos, s)
        wins += (len(pos) - hi) + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def sweep(frames):
    frames = sorted(frames)  # by score asc (low risk first)
    n = len(frames)
    n_bad = sum(1 for _, bad, _ in frames if bad)
    out = {'n': n, 'base_bad_rate': n_bad / n if n else float('nan'),
           'auroc': auroc(frames)}
    best = {}
    kept_bad = 0
    for i, (score, bad, _) in enumerate(frames, start=1):
        kept_bad += bad
        cov = i / n
        abr = kept_bad / i
        for gate in (0.0, 0.05, 0.10, 0.20):
            if abr <= gate:
                key = f'abr<={gate:.2f}'
                if key not in best or cov > best[key][0]:
                    best[key] = (cov, abr, score, i)
    for gate in (0.0, 0.05, 0.10, 0.20):
        key = f'abr<={gate:.2f}'
        if key in best:
            cov, abr, tau, kept = best[key]
            out[key] = f'cov={cov:.1%} abr={abr:.1%} tau={tau:.4f} kept={kept}'
        else:
            out[key] = 'infeasible'
    return out


def main():
    print(f'{"dataset":<12} {"n_acc":>6} {"bad%":>6} {"AUROC":>6}')
    results = {}
    for ds, thr in DATASETS.items():
        frames = load(ds, thr)
        r = sweep(frames)
        results[ds] = r
        print(f'{ds:<12} {r["n"]:>6} {r["base_bad_rate"]:>6.1%} {r["auroc"]:>6.3f}')
        for gate in ('abr<=0.00', 'abr<=0.05', 'abr<=0.10', 'abr<=0.20'):
            print(f'    max-coverage @ {gate}: {r[gate]}')

    with open('/tmp/caves_per_dataset_operating_points.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'n_accepted', 'base_bad_rate', 'auroc',
                    'cov_at_abr0', 'cov_at_abr5', 'cov_at_abr10', 'cov_at_abr20'])
        for ds, r in results.items():
            w.writerow([ds, r['n'], f"{r['base_bad_rate']:.4f}", f"{r['auroc']:.4f}",
                        r['abr<=0.00'], r['abr<=0.05'], r['abr<=0.10'], r['abr<=0.20']])
    print('Saved /tmp/caves_per_dataset_operating_points.csv')


if __name__ == '__main__':
    main()
