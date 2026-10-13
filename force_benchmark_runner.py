"""Stream the complete force benchmarks into resumable Cartesian-force shards.

Graph preparation and GPU inference are batched; public FP-DeErr force math
is used without modification. No dataset sampling is implied by partial runs.
"""
import argparse
from collections import deque
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import itertools
import json
import os
for _thread_name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_thread_name]='1'
from pathlib import Path
import sys
import time
import datetime
import gc
import reproduce
from relaxation_runner import atomic_json

OUT = reproduce.OUT / 'force_benchmarks'
CONVERTER = None
ADAPTOR = None
WEIGHTS = reproduce.ROOT / '.venv/Lib/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar'


def init_graph_worker():
    global CONVERTER, ADAPTOR
    import torch
    from chgnet.graph.converter import CrystalGraphConverter
    from pymatgen.io.ase import AseAtomsAdaptor
    torch.set_num_threads(1)
    CONVERTER = CrystalGraphConverter(atom_graph_cutoff=6, bond_graph_cutoff=3, algorithm='legacy')
    CONVERTER.set_isolated_atom_response('warn')  # CHGNetCalculator's upstream default
    ADAPTOR = AseAtomsAdaptor()


def prepare_graphs(batch):
    from pymatgen.core import Structure
    # Match the ASE calculator's Structure -> Atoms -> Structure conversion.
    graphs = []
    for row in batch:
        s = ADAPTOR.get_structure(ADAPTOR.get_atoms(Structure.from_dict(row['structure'])))
        graph = CONVERTER(s)
        from graph_shape_adapter import normalize_empty_atom_graph
        normalize_empty_atom_graph(graph)
        # NumPy serialization avoids the expensive per-tensor Windows IPC
        # handles while preserving every graph tensor's values and dtype.
        graphs.append({key: value.detach().numpy() if hasattr(value, 'detach') else value
                       for key, value in graph.to_dict().items()})
    return batch, graphs


def structures(name, path, start=0):
    if name == 'matpes_pbe':
        with Path(path).open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                row = json.loads(line)
                yield {'index': i, 'source_id': row['matpes_id'], 'structure': row['structure'], 'forces': row['forces']}
    else:
        from ase_db_backends.aselmdb import LMDBDatabase
        from pymatgen.io.ase import AseAtomsAdaptor
        db = LMDBDatabase(path, readonly=True)
        for i, row in enumerate(db.select()):
            if i < start:
                continue
            yield {'index': i, 'source_id': str(row.id), 'structure': AseAtomsAdaptor.get_structure(row.toatoms()).as_dict(), 'forces': row.forces}


def batches(iterator, size):
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def predict(model, graphs, size, cleanup_cuda=False, singleton_disk_offload=False,
            retry_events=None):
    import torch
    results = []
    # These are exactly the same contiguous mini-batches used internally by
    # predict_graph. An OOM retries only the affected mini-batch, retaining
    # already-computed predictions rather than recalculating the whole shard.
    for offset in range(0, len(graphs), size):
        chunk = graphs[offset:offset + size]
        pending = deque([(offset, chunk, False)])
        while pending:
            graph_offset, part, offload = pending.popleft()
            retry_needed = False
            try:
                # One actual forward call at a time. predict_graph's internal
                # loop otherwise retains the previous raw derivative graph
                # while evaluating the next smaller OOM-retry batch.
                if offload:
                    from cuda_saved_tensor_offload import saved_tensors_on_disk
                    context = saved_tensors_on_disk(reproduce.ROOT/'tmp/cuda_offload')
                else:
                    context = nullcontext(None)
                with context as offload_statistics:
                    result = model.predict_graph(part, task='efsm', return_crystal_feas=True,
                                                 batch_size=len(part))
                if offload:
                    import numpy as np
                    assert all(np.isfinite(np.asarray(value)).all() for value in result.values())
                results.extend([result] if len(part) == 1 else result)
                if offload:
                    event = {'event': 'cuda_singleton_disk_offload',
                             'graph_offset': graph_offset, **offload_statistics}
                    if retry_events is not None:
                        retry_events.append(event)
                    print(json.dumps(event), flush=True)
            except torch.cuda.OutOfMemoryError:
                if len(part) == 1:
                    if not singleton_disk_offload or offload:
                        raise
                    # Retry outside the handler, after its derivative graph is
                    # released. Arithmetic, precision, and device stay unchanged.
                    retry_needed = True
                    pending.appendleft((graph_offset, part, True))
                    print(json.dumps({'event': 'cuda_singleton_offload_retry',
                                      'graph_offset': graph_offset}), flush=True)
                else:
                    split = max(1, len(part)//2)
                    retry_needed = True
                    # Left-first ordering retains already-successful predictions.
                    pending.appendleft((graph_offset+split, part[split:], False))
                    pending.appendleft((graph_offset, part[:split], False))
                    print(json.dumps({'event':'cuda_oom_retry','graph_offset':graph_offset,
                                      'chunk_graphs':len(part),'retry_batch_size':split}), flush=True)
            # Leave the exception handler before freeing memory: its traceback
            # otherwise keeps the failed forward's tensors alive during cleanup.
            if retry_needed or cleanup_cuda:
                gc.collect()
                torch.cuda.empty_cache()
    return results


def run(args):
    import numpy as np
    import torch
    from chgnet.model.model import CHGNet
    from chgnet.graph.converter import CrystalGraphConverter
    sys.path.insert(0, str(reproduce.REPO / 'Force_error/scripts'))
    from force_results import build_force_results
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable; no silent CPU fallback'
        if args.cuda_memory_fraction is not None:
            filename = 'gpu_memory_cleanup_validation.json' if args.cuda_memory_fraction == 0.75 else f'gpu_memory_cleanup_validation_{args.cuda_memory_fraction:g}.json'
            allocator_validation_path = reproduce.OUT/filename
            validation=json.loads(allocator_validation_path.read_text())
            assert validation['validation_pass'] and validation['memory_fraction']==args.cuda_memory_fraction
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    manifest = json.loads((reproduce.ROOT / 'datasets/manifest.json').read_text(encoding='utf-8'))
    dataset = manifest[args.dataset]
    assert reproduce.sha(WEIGHTS) == 'd14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1'
    model = CHGNet.from_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=False)['model'])
    model.graph_converter = CrystalGraphConverter(atom_graph_cutoff=6, bond_graph_cutoff=3, algorithm='legacy')
    model.eval().to(args.device)
    execution = {'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'graph_workers':args.graph_workers,'retry_scope':'mini_batch',
                 'prefetch_shards':args.prefetch_shards or max(2,args.graph_workers),
                 'inference_optimized':args.inference_optimized}
    if args.cuda_memory_fraction is not None:
        execution['cuda_memory_fraction']=args.cuda_memory_fraction
        execution['release_unused_cache_after_mini_batch_and_shard']=True
        execution['allocator_validation_sha256']=reproduce.sha(allocator_validation_path)
    if args.singleton_disk_offload:
        assert args.device == 'cuda' and args.cuda_memory_fraction == 0.9
        validation_path = reproduce.OUT/'chgnet_cuda_disk_offload_validation.json'
        validation = json.loads(validation_path.read_text())
        helper = reproduce.ROOT/'cuda_saved_tensor_offload.py'
        assert validation['validation_pass']
        assert validation['weights_sha256'] == reproduce.sha(WEIGHTS)
        assert validation['implementation_sha256'] == reproduce.sha(helper)
        assert validation['device'] == 'cuda' and validation['dtype'] == 'float32'
        assert validation['dense_case']['original_singleton_oom']
        assert validation['dense_case']['all_fields_finite']
        assert {c['batch_size'] for c in validation['checks'] if c['deterministic']} == {1, 16}
        assert all(v['equal'] and v['finite'] for c in validation['checks'] if c['deterministic']
                   for v in c['offload_compare'].values())
        execution['singleton_disk_offload'] = {
            'validation_sha256': reproduce.sha(validation_path),
            'implementation_sha256': reproduce.sha(helper),
            'trigger': 'CUDA singleton OOM only', 'arithmetic_device': 'cuda'}
        controls_path = reproduce.OUT/'chgnet_disk_offload_retry_validation.json'
        controls = json.loads(controls_path.read_text())
        assert controls['validation_pass'] and controls['runner_sha256'] == reproduce.sha(Path(__file__))
        execution['singleton_disk_offload']['retry_controls_sha256'] = reproduce.sha(controls_path)
    shape_validation_path = reproduce.OUT/'empty_graph_validation.json'
    shape_validation = json.loads(shape_validation_path.read_text())
    assert shape_validation['validation_pass']
    assert shape_validation['weights_sha256'] == reproduce.sha(WEIGHTS)
    assert shape_validation['adapter_sha256'] == reproduce.sha(reproduce.ROOT/'graph_shape_adapter.py')
    execution['empty_graph_schema_adapter'] = {
        'validation_sha256': reproduce.sha(shape_validation_path),
        'implementation_sha256': shape_validation['adapter_sha256']}
    partition_validation=reproduce.OUT/'inference_optimization_validation_cuda_retry_only_deterministic.json'
    if args.device=='cuda' and partition_validation.exists():
        assert json.loads(partition_validation.read_text())['all_outputs_exactly_equal']
        execution['partition_validation_sha256']=reproduce.sha(partition_validation)
    if args.inference_optimized:
        validation_path=reproduce.OUT/('inference_optimization_validation_cuda_deterministic.json'
            if args.device=='cuda' else 'inference_optimization_validation_cpu.json')
        validation=json.loads(validation_path.read_text())
        assert validation['all_outputs_exactly_equal'], 'Inference optimization has not passed exact validation'
        assert validation['weights_sha256']==reproduce.sha(WEIGHTS)
        from inference_optimization import enable
        execution['optimization']=enable(model)
        execution['validation_sha256']=reproduce.sha(validation_path)
    provenance = {'dataset': dataset, 'weights_sha256': reproduce.sha(WEIGHTS),
        'repository_commit': '0d5d9a62d0ef3b7ce4da58620a7919e0e49ba815',
        'device': args.device, 'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'dtype': 'float32', 'tf32': False, 'graph_algorithm': 'legacy', 'atom_cutoff': 6, 'bond_cutoff': 3,
        'task': 'efsm', 'prediction_batch_size': args.batch_size, 'shard_size': args.shard_size,
        'force_results_sha256': reproduce.sha(reproduce.REPO / 'Force_error/scripts/force_results.py'),
        'versions': {n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'pymatgen', 'ase', 'chgnet']}}
    folder = OUT / args.dataset / (args.run_name or args.device)
    shards = folder / 'shards'
    shards.mkdir(parents=True, exist_ok=True)
    provenance_path = folder / 'provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text(encoding='utf-8')) == provenance
    else:
        atomic_json(provenance_path, provenance)
    planned = min(args.limit or dataset['structures'], dataset['structures'])
    begun = time.perf_counter()
    done = atoms_done = 0
    # Verify the contiguous saved prefix before streaming new inputs. Resume
    # skips JSON decoding and Structure/Atoms conversion for cached records.
    for meta_path in sorted(shards.glob('*.json')):
        meta=json.loads(meta_path.read_text())
        if meta['first']>=planned:
            break
        assert meta['first']==done and meta['last']<planned
        assert reproduce.sha(meta_path.with_suffix('.npz'))==meta['sha256']
        done=meta['last']+1
        atoms_done+=meta['atoms']
    source = itertools.islice(structures(args.dataset, dataset['path'],start=done),planned-done)
    execution['resumed_after_verified_structures']=done
    queue = deque()
    execution_path=folder/'execution_history.json'
    history=json.loads(execution_path.read_text()) if execution_path.exists() else []
    history.append(execution)
    atomic_json(execution_path,history)
    timing_totals = {key:0.0 for key in ['graph_wait_seconds','host_conversion_seconds','inference_seconds','save_seconds']}
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '1'

    def save(batch, graphs):
        shard_timing={}
        from chgnet.graph.crystalgraph import CrystalGraph
        stage=time.perf_counter()
        graphs = [CrystalGraph(**{key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                                  for key, value in graph.items()}) for graph in graphs]
        shard_timing['host_conversion_seconds']=time.perf_counter()-stage
        stage=time.perf_counter()
        retry_events = []
        predictions = predict(model, graphs, args.batch_size, cleanup_cuda=args.cuda_memory_fraction is not None,
                              singleton_disk_offload=args.singleton_disk_offload, retry_events=retry_events)
        shard_timing['inference_seconds']=time.perf_counter()-stage
        stage=time.perf_counter()
        assert len(predictions) == len(batch)
        dft = [np.asarray(row['forces'], dtype=float) for row in batch]
        fp = [np.asarray(pred['f'], dtype=float) for pred in predictions]
        for d, p in zip(dft, fp):
            assert d.shape == p.shape and np.isfinite(d).all() and np.isfinite(p).all()
        standard = build_force_results(dft, {'CHGNet': fp}, structure_ids=[r['index'] for r in batch])['CHGNet']
        packed = {key: np.asarray(value) for key, value in standard.items()}
        packed['structure_id'] = packed['structure_id'].astype(np.int64)
        packed.update(dft_cartesian=np.concatenate(dft), fp_cartesian=np.concatenate(fp),
            structure_indices=np.array([r['index'] for r in batch]),
            source_ids=np.array([str(r['source_id']) for r in batch]),
            offsets=np.cumsum([0] + [len(f) for f in dft]),
            energy_eV_per_atom=np.array([p['e'] for p in predictions]))
        key = f"{batch[0]['index']:07d}"
        destination = shards / (key + '.npz')
        with destination.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, **packed)
        destination.with_suffix('.tmp').replace(destination)
        atomic_json(shards / (key + '.json'), {'first': batch[0]['index'], 'last': batch[-1]['index'],
            'structures': len(batch), 'atoms': len(packed['dft_force_magnitude']),
            'sha256': reproduce.sha(destination), 'undefined_force_angles': int(np.isnan(packed['force_angle_error']).sum()),
            'execution':execution,'timing':shard_timing,
            'singleton_disk_offloads': [{**event, 'structure_index': batch[event['graph_offset']]['index'],
                                        'source_id': batch[event['graph_offset']]['source_id']}
                                       for event in retry_events],
            'empty_edge_structure_indices': [row['index'] for row, graph in zip(batch, graphs) if graph.atom_graph.numel() == 0]})
        shard_timing['save_seconds']=time.perf_counter()-stage
        for key,value in shard_timing.items():
            timing_totals[key]+=value
        return len(batch), len(packed['dft_force_magnitude'])

    def consume(future):
        stage=time.perf_counter()
        prepared=future.result()
        timing_totals['graph_wait_seconds']+=time.perf_counter()-stage
        return save(*prepared)

    def progress(status_override=None, log=True):
        elapsed = time.perf_counter() - begun
        record = {'dataset': args.dataset, 'device': args.device, 'completed_structures': done,
            'planned_structures': planned, 'dataset_structures': dataset['structures'],
            'completed_atoms': atoms_done, 'elapsed_seconds': elapsed,
            'full_dataset_complete': done == dataset['structures'], 'status': status_override or ('complete' if done == planned else 'running'),
            'session_timing':timing_totals,'execution':execution}
        if args.device=='cuda':
            record['cuda_memory']={'allocated_bytes':torch.cuda.memory_allocated(),
                'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            try:
                import psutil
                record['host_rss_bytes']=psutil.Process().memory_info().rss
            except ImportError:
                pass
        atomic_json(folder / 'progress.json', record)
        if log:
            print(json.dumps(record), flush=True)

    progress()
    with ProcessPoolExecutor(max_workers=args.graph_workers, initializer=init_graph_worker) as pool:
        for batch in batches(source, args.shard_size):
            key = f"{batch[0]['index']:07d}"
            meta_path = shards / (key + '.json')
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                assert meta['first'] == batch[0]['index'] and meta['last'] == batch[-1]['index']
                assert reproduce.sha(shards / (key + '.npz')) == meta['sha256']
                done += meta['structures']
                atoms_done += meta['atoms']
                continue
            queue.append(pool.submit(prepare_graphs, batch))
            if len(queue) >= execution['prefetch_shards']:
                n, a = consume(queue.popleft())
                done += n
                atoms_done += a
                if args.cuda_memory_fraction is not None:
                    gc.collect()
                    torch.cuda.empty_cache()
                progress(log=done % (args.shard_size * 10) == 0)
                if (folder/'stop_after_shard').exists():
                    progress(status_override='paused_after_shard')
                    print('Stopped at a verified shard boundary by request',flush=True)
                    return
        while queue:
            n, a = consume(queue.popleft())
            done += n
            atoms_done += a
            if args.cuda_memory_fraction is not None:
                gc.collect()
                torch.cuda.empty_cache()
            progress()
    assert done == planned
    progress()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['matpes_pbe', 'omat24'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--graph-workers', type=int, default=4)
    parser.add_argument('--run-name')
    parser.add_argument('--inference-optimized',action='store_true')
    parser.add_argument('--prefetch-shards',type=int)
    parser.add_argument('--cuda-memory-fraction',type=float)
    parser.add_argument('--singleton-disk-offload', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        progress_path = OUT / args.dataset / (args.run_name or args.device) / 'progress.json'
        if progress_path.exists():
            failed_progress = json.loads(progress_path.read_text(encoding='utf-8'))
            failed_progress.update(status='failed', error=repr(error))
            atomic_json(progress_path, failed_progress)
        raise
