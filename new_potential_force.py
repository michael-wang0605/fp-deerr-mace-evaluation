"""Stream full force populations through validated float64 MACE inference."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import datetime
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import time
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import reproduce
import force_benchmark_runner as sources
import new_potential_static as mace
from relaxation_runner import atomic_json

NAME = 'mace_omat_0_small_cpu'


def initialize(device='cpu'):
    mace.initialize(device)
    if device == 'cuda':
        import torch
        torch.cuda.set_per_process_memory_fraction(.9)


def predict_cuda(atoms, batch_size):
    """Retry only failed contiguous partitions, retaining completed results."""
    import gc
    import torch
    from mace_batch_inference import predict
    results, events = [], []
    for offset in range(0,len(atoms),batch_size):
        pending = deque([(offset,atoms[offset:offset+batch_size])])
        while pending:
            first, part = pending.popleft()
            retry = False
            try:
                results.extend(predict(mace.CALC,part,batch_size=len(part)))
            except RuntimeError as error:
                # TorchScript wraps CUDA allocation failures in RuntimeError.
                # Propagate every unrelated runtime error unchanged.
                wrapped_cuda_oom = ('The following operation failed in the TorchScript interpreter.' in str(error)
                    and 'RuntimeError: CUDA out of memory.' in str(error))
                if not isinstance(error, torch.cuda.OutOfMemoryError) and not wrapped_cuda_oom:
                    raise
                if len(part)==1:
                    raise
                split = len(part)//2
                pending.appendleft((first+split,part[split:]))
                pending.appendleft((first,part[:split]))
                events.append({'offset':first,'structures':len(part),'retry_batch_size':split})
                retry = True
            if retry:
                # Run outside the handler so traceback-owned tensors are gone.
                gc.collect()
                torch.cuda.empty_cache()
    assert len(results)==len(atoms)
    return results, events


def compute(batch, destination, expected, batch_size):
    from pymatgen.core import Structure
    from mace_batch_inference import predict
    sys.path.insert(0, str(reproduce.REPO/'Force_error/scripts'))
    from force_results import build_force_results
    begun = time.perf_counter()
    atoms = [mace.ADAPTOR.get_atoms(Structure.from_dict(row['structure'])) for row in batch]
    cuda = str(mace.CALC.device).startswith('cuda')
    if cuda:
        predictions, retry_events = predict_cuda(atoms,batch_size)
    else:
        predictions = predict(mace.CALC, atoms, batch_size=batch_size)
    dft = [np.asarray(row['forces'], dtype=float) for row in batch]
    fp = [prediction['forces'] for prediction in predictions]
    for a, b in zip(dft, fp):
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    standard = build_force_results(dft, {mace.KEY: fp}, structure_ids=[row['index'] for row in batch])[mace.KEY]
    packed = {key: np.asarray(value) for key, value in standard.items()}
    packed['structure_id'] = packed['structure_id'].astype(np.int64)
    packed.update(dft_cartesian=np.concatenate(dft), fp_cartesian=np.concatenate(fp),
        structure_indices=np.array([row['index'] for row in batch]), source_ids=np.array([str(row['source_id']) for row in batch]),
        offsets=np.cumsum([0]+[len(force) for force in dft]),
        energy_eV_per_atom=np.array([prediction['energy_total']/len(item) for prediction, item in zip(predictions, atoms)]))
    assert np.isfinite(packed['energy_eV_per_atom']).all()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.tmp').open('wb') as output:
        np.savez_compressed(output, **packed)
    destination.with_suffix('.tmp').replace(destination)
    metadata = {'first': batch[0]['index'], 'last': batch[-1]['index'], 'structures': len(batch),
        'atoms': len(packed['dft_force_magnitude']), 'sha256': reproduce.sha(destination),
        'undefined_force_angles': int(np.isnan(packed['force_angle_error']).sum()),
        'provenance_sha256': expected, 'seconds': time.perf_counter()-begun}
    if cuda:
        metadata['cuda_oom_retries'] = retry_events
    atomic_json(destination.with_suffix('.json'), metadata)
    return metadata


def run(args):
    manifest = json.loads((reproduce.ROOT/'datasets/manifest.json').read_text())
    dataset = manifest[args.dataset]
    validation_path = reproduce.OUT/'mace_batch_validation.json'
    validation = json.loads(validation_path.read_text())
    assert validation['validation_pass'] and validation['weights_sha256']==mace.MODEL_SHA
    assert validation['implementation_sha256']==reproduce.sha(reproduce.ROOT/'mace_batch_inference.py')
    assert args.batch_size in [trial['batch_size'] for trial in validation['trials']]
    if args.device == 'cuda':
        assert args.workers == 1, 'Only one GPU inference worker may own this output'
        gpu_validation_path = reproduce.OUT/'mace_cpu_cuda_validation.json'
        gpu_validation = json.loads(gpu_validation_path.read_text())
        assert gpu_validation['validation_pass'] and gpu_validation['dtype']=='float64'
        assert gpu_validation['weights_sha256']==mace.MODEL_SHA
        assert gpu_validation['implementation_sha256']==reproduce.sha(reproduce.ROOT/'mace_batch_inference.py')
        assert gpu_validation['batch_size']==args.batch_size
        for name, version in gpu_validation['versions']['cuda'].items():
            assert importlib.metadata.version(name)==version, (name,version)
        partition_validation_path = reproduce.OUT/'mace_cuda_partition_validation.json'
        partition_validation = json.loads(partition_validation_path.read_text())
        assert partition_validation['validation_pass'] and partition_validation['weights_sha256']==mace.MODEL_SHA
        assert partition_validation['implementation_sha256']==reproduce.sha(reproduce.ROOT/'mace_batch_inference.py')
        assert {row['batch_size'] for row in partition_validation['trials']}=={1,2,4,8,16}
        retry_control_path = reproduce.OUT/'mace_cuda_retry_control_validation.json'
        retry_control = json.loads(retry_control_path.read_text())
        assert retry_control['validation_pass']
        assert retry_control['runner_sha256']==reproduce.sha(reproduce.ROOT/'new_potential_force.py')
    names = ['mace-torch', 'torch', 'e3nn', 'numpy', 'ase', 'pymatgen', 'matscipy']
    provenance = {'model': 'MACE-OMAT-0 Small', 'dataset': dataset, 'weights_sha256': mace.MODEL_SHA,
        'repository_commit': '0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815',
        'device': args.device, 'dtype': 'float64', 'head': 'omat_pbe', 'cutoff_angstrom': 6.0, 'dispersion_correction': False,
        'batch_size': args.batch_size, 'shard_size': 128, 'validation_sha256': reproduce.sha(validation_path),
        'force_results_sha256': reproduce.sha(reproduce.REPO/'Force_error/scripts/force_results.py'),
        'versions': {name: importlib.metadata.version(name) for name in names}}
    if args.device == 'cuda':
        provenance['cpu_cuda_validation_sha256'] = reproduce.sha(gpu_validation_path)
        provenance['cuda_partition_validation_sha256'] = reproduce.sha(partition_validation_path)
        provenance['cuda_retry_control_validation_sha256'] = reproduce.sha(retry_control_path)
        provenance['cuda_memory_fraction'] = .9
        provenance['tf32'] = False
    run_name = f'mace_omat_0_small_{args.device}'
    folder = reproduce.OUT/'force_benchmarks'/args.dataset/run_name
    provenance_path = folder/'provenance.json'
    if provenance_path.exists():
        retained_provenance = json.loads(provenance_path.read_text())
        if retained_provenance != provenance and args.device == 'cuda':
            # Retry validation describes execution controls, not model math.
            # Preserve the original provenance and every cached shard verbatim;
            # record the newly validated control implementation per execution.
            previous_digest = retained_provenance['cuda_retry_control_validation_sha256']
            archive = reproduce.OUT/f'mace_cuda_retry_control_validation_{previous_digest}.json'
            assert archive.exists() and reproduce.sha(archive) == previous_digest
            assert json.loads(archive.read_text())['validation_pass'] is True
            compatible = dict(provenance)
            compatible['cuda_retry_control_validation_sha256'] = previous_digest
            assert retained_provenance == compatible, 'Scientific provenance changed; cannot reuse cached results'
        else:
            assert retained_provenance == provenance
        provenance = retained_provenance
    else:
        atomic_json(provenance_path, provenance)
    planned = min(args.limit or dataset['structures'], dataset['structures'])
    done = atoms_done = 0
    cached = {}
    for path in sorted((folder/'shards').glob('*.json')):
        record = json.loads(path.read_text())
        assert record['first']%128==0 and record['last']==min(record['first']+127,planned-1)
        assert record['structures']==record['last']-record['first']+1
        assert record['sha256']==reproduce.sha(path.with_suffix('.npz'))
        assert record['provenance_sha256']==reproduce.sha(provenance_path)
        assert record['first'] not in cached
        cached[record['first']]=record
    while done in cached:
        record=cached[done]
        done=record['last']+1
        atoms_done+=record['atoms']
    begun = time.perf_counter()
    execution = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'workers': args.workers, 'threads_per_worker': 1, 'resumed_from': done}
    if args.device == 'cuda':
        execution.update(runner_sha256=reproduce.sha(reproduce.ROOT/'new_potential_force.py'),
            cuda_retry_control_validation_sha256=reproduce.sha(retry_control_path),
            retained_provenance_retry_validation_sha256=provenance['cuda_retry_control_validation_sha256'])
    history_path = folder/'execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(execution)
    atomic_json(history_path, history)

    def progress(status=None, last=None):
        record = {'dataset': args.dataset, 'device': args.device, 'completed_structures': done,
            'planned_structures': planned, 'dataset_structures': dataset['structures'], 'completed_atoms': atoms_done,
            'full_dataset_complete': done==dataset['structures'], 'status': status or ('complete' if done==planned else 'running'),
            'elapsed_seconds': time.perf_counter()-begun, 'execution': execution, 'last': last}
        atomic_json(folder/'progress.json', record)
        if done%1280==0 or status:
            print(json.dumps(record), flush=True)
    progress()
    queue = deque()
    def consume(record):
        nonlocal done, atoms_done
        assert record['first']==done
        done+=record['structures']; atoms_done+=record['atoms']
        progress(last=record)
    try:
        source = itertools.islice(sources.structures(args.dataset, dataset['path'], start=done), planned-done)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.device,)) as pool:
            for batch in sources.batches(source, 128):
                if batch[0]['index'] in cached:
                    # Interrupted parallel workers can leave verified shards
                    # beyond a gap. Fill missing inputs and retain every saved
                    # shard without recalculating or overwriting it.
                    while queue:
                        consume(queue.popleft().result())
                    consume(cached[batch[0]['index']])
                    continue
                destination = folder/'shards'/f"{batch[0]['index']:07d}.npz"
                queue.append(pool.submit(compute, batch, str(destination), reproduce.sha(provenance_path), args.batch_size))
                if len(queue)>=max(2, args.workers*2):
                    consume(queue.popleft().result())
                    if (folder/'stop_after_shard').exists():
                        while queue:
                            consume(queue.popleft().result())
                        progress(status='paused_after_shard')
                        return
            while queue:
                consume(queue.popleft().result())
    except Exception as error:
        progress(status='failed', last={'error': repr(error)})
        raise
    assert done==planned
    progress()
    if not args.limit:
        from force_benchmark_analysis import analyze
        analyze(args.dataset, args.device, run_name=run_name, model_label='MACE-OMAT-0 Small', compare_published=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['matpes_pbe', 'omat24'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    run(parser.parse_args())
