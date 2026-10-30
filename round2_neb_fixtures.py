"""Compute owned CPU NEB fixtures with the unchanged benchmark template."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def worker_setup():
    import new_potential_neb as base
    base.initialize('full')


def main(args):
    folder = ROOT / 'outputs/round2/adapters_v2' / args.model
    sys.path.insert(0, str(folder))
    import new_potential_neb as base
    import neb_full_runner as full
    import reproduce
    from relaxation_runner import atomic_json
    reference = reproduce.load(base.REFERENCE)
    profile = base.provenance('full')
    jobs = []
    for key in args.pathway:
        assert key in reference['pathways']
        path = base.record_path('full', key)
        if path.exists():
            record = json.loads(path.read_text())
            assert record['id'] == key and record['provenance'] == profile
            assert record['result']['neb_status']['calculation_status'] != 'failed'
        else:
            jobs.append((key, reference['pathways'][key]))
    atomic_json(base.folder('full') / 'fixtures_plan.json', {
        'pathways': args.pathway, 'provenance': profile,
        'workers': args.workers, 'template_sha256': profile['template_sha256']})
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_setup) as pool:
        futures = [pool.submit(full.compute, job, profile) for job in jobs]
        for future in futures:
            future.result()
    results = []
    for key in args.pathway:
        path = base.record_path('full', key)
        record = json.loads(path.read_text())
        assert record['provenance'] == profile and record['id'] == key
        assert record['result']['neb_status']['calculation_status'] != 'failed'
        results.append({'pathway': key, 'sha256': reproduce.sha(path),
                        'converged': record['result']['neb_status']['neb_converged']})
    atomic_json(base.folder('full') / 'fixtures_validation.json', {'pass': True, 'records': results})
    print(json.dumps({'model': args.model, 'fixtures': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--pathway', action='append', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    # Spawned Windows workers re-import sibling adapters from this same path.
    sys.path.insert(0, str(ROOT / 'outputs/round2/adapters_v2' / args.model))
    main(args)
