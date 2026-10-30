"""Package audited results for the repository's email submission workflow.

Reads retained scientific outputs; writes only a separate submission directory.
"""
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import prepare_upstream_leaderboard as builder
import force_benchmark_analysis as analysis

ROOT = builder.ROOT
DEST = builder.OUT / 'submission_assets'
FIELDS = builder.FIELDS + ['structure_id', 'atom_index']


def compressed_copy(source, target):
    with source.open('rb') as src, gzip.GzipFile(filename='', mode='wb', fileobj=target.open('wb'), mtime=0, compresslevel=1) as out:
        shutil.copyfileobj(src, out)
    with gzip.open(target, 'rb') as src:
        digest = hashlib.sha256()
        for chunk in iter(lambda: src.read(1 << 20), b''):
            digest.update(chunk)
    assert digest.hexdigest() == builder.sha(source)


def write_array(stream, array):
    stream.write('[')
    for first in range(0, len(array), 65536):
        if first:
            stream.write(',')
        stream.write(json.dumps(array[first:first+65536].tolist(), separators=(',', ':'), allow_nan=True)[1:-1])
    stream.write(']')


def force_dataset(dataset, models):
    path = DEST / (dataset + '_force_results_standardized.json.gz')
    evidence = {}
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1, newline='\n') as out:
        out.write('{"schema_version":"1.0","dataset_name":' + json.dumps(dataset))
        out.write(',"units":{"force":"eV/Angstrom","angle":"degree"},"models":{')
        for model_index, (key, model) in enumerate(models.items()):
            folder = ROOT / model['force_folders'][dataset]
            summary = builder.read(folder / 'summary.json')
            parts = {name: [] for name in FIELDS}
            shard_hashes = {}
            index = atoms = 0
            for shard_path in sorted((folder / 'shards').glob('*.npz')):
                meta = builder.read(shard_path.with_suffix('.json'))
                assert meta['first'] == index and builder.sha(shard_path) == meta['sha256']
                with np.load(shard_path, allow_pickle=False) as shard:
                    assert np.array_equal(shard['structure_indices'], np.arange(meta['first'], meta['last'] + 1))
                    assert len(shard['atom_index']) == meta['atoms']
                    for name in FIELDS:
                        parts[name].append(shard[name].copy())
                shard_hashes[shard_path.name] = meta['sha256']
                index = meta['last'] + 1
                atoms += meta['atoms']
            arrays = {name: np.concatenate(value) for name, value in parts.items()}
            del parts
            assert index == summary['structures'] and atoms == summary['atoms']
            table, _, _, _, finite, _ = analysis.summary_from_arrays(dataset, {name: arrays[name] for name in builder.FIELDS}, model['display_name'])
            assert table.loc[model['display_name']].to_dict() == summary['summary_metrics']
            if model_index:
                out.write(',')
            out.write(json.dumps(model['display_name']) + ':{')
            for field_index, name in enumerate(FIELDS):
                if field_index:
                    out.write(',')
                out.write(json.dumps(name) + ':')
                write_array(out, arrays[name])
            out.write('}')
            evidence[key] = {'structures': index, 'atoms': atoms, 'undefined_angles_retained': int((~finite).sum()),
                             'summary_sha256': builder.sha(folder / 'summary.json'),
                             'source_shards_sha256': shard_hashes,
                             'array_bytes_sha256': {name: hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in FIELDS},
                             'array_dtypes': {name: str(arrays[name].dtype) for name in FIELDS}}
            print(json.dumps({'exported_force_model': key, 'dataset': dataset, 'atoms': atoms}), flush=True)
            del arrays, table
            gc.collect()
        out.write('}}\n')
    # Check actual JSON-decoded values and ordering against every saved array.
    with gzip.open(path, 'rt', encoding='utf-8') as src:
        decoded = json.load(src)
    for key, model in models.items():
        row = decoded['models'][model['display_name']]
        for name in FIELDS:
            array = np.asarray(row[name], dtype=evidence[key]['array_dtypes'][name])
            assert len(array) == evidence[key]['atoms']
            assert hashlib.sha256(array.tobytes()).hexdigest() == evidence[key]['array_bytes_sha256'][name]
    del decoded
    gc.collect()
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': builder.sha(path), 'models': evidence,
            'decoded_arrays_equal_audited_shards': True,
            'undefined_angles': 'All atoms retained; undefined angles use NaN, supported by the upstream Python JSON schema.'}


def main():
    scope, models = builder.specs(False)
    audits = builder.gate(scope, models)
    DEST.mkdir(parents=True, exist_ok=True)
    files = {}
    for dataset in ['matpes_pbe', 'omat24']:
        proof_path = DEST / (dataset + '_export_validation.json')
        if proof_path.exists():
            proof = builder.read(proof_path)
            assert builder.sha(DEST / proof['file']) == proof['sha256']
            assert all(builder.sha(ROOT / models[key]['force_folders'][dataset] / 'summary.json') == row['summary_sha256'] for key, row in proof['models'].items())
        else:
            proof = force_dataset(dataset, models)
            proof_path.write_text(json.dumps(proof, indent=2), encoding='utf-8')
        files[proof['file']] = {'bytes': proof['bytes'], 'sha256': proof['sha256'], 'component': 'force', 'dataset': dataset}
    for key in models:
        for stage, source in [('static_phase', ROOT / 'outputs' / key / 'standardized_results.json'),
                              ('relaxed_phase', ROOT / 'outputs' / key / 'relaxation/standardized_results.json')]:
            target = DEST / (key + '_' + stage + '_standardized.json.gz')
            compressed_copy(source, target)
            files[target.name] = {'bytes': target.stat().st_size, 'sha256': builder.sha(target),
                                  'uncompressed_sha256': builder.sha(source), 'component': stage, 'model': key}
    candidate = builder.OUT / 'upstream_contribution'
    manifest = builder.read(candidate / 'manifest.json')
    for relative in manifest['files_sha256']:
        if relative.endswith('.gz'):
            source = candidate / relative
            target = DEST / (source.parent.name + '_' + source.name)
            shutil.copy2(source, target)
            assert builder.sha(target) == manifest['files_sha256'][relative]
            files[target.name] = {'bytes': target.stat().st_size, 'sha256': builder.sha(target), 'component': 'NEB'}
    review = DEST / 'review_bundle.zip'
    shutil.make_archive(str(review.with_suffix('')), 'zip', builder.OUT / 'review_bundle')
    files[review.name] = {'bytes': review.stat().st_size, 'sha256': builder.sha(review), 'component': 'tables_and_audits'}
    result = {'pass': True, 'models': list(models), 'audit_sha256': {key: builder.sha(ROOT / p) for key, p in audits.items()},
              'files': files, 'packager_sha256': builder.sha(Path(__file__)), 'scientific_outputs_modified': False}
    (DEST / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (DEST / 'SHA256SUMS.txt').write_text(''.join(f"{value['sha256']}  {name}\n" for name, value in files.items()), encoding='utf-8')
    print(json.dumps({'pass': True, 'assets': len(files), 'total_bytes': sum(x['bytes'] for x in files.values())}), flush=True)


if __name__ == '__main__':
    main()
