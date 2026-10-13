"""Validate exact CUDA outputs and the formerly failing dense MatPES input."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import gc
import itertools
import json
import threading
import time
import numpy as np
import psutil
import torch
from chgnet.model.model import CHGNet
from chgnet.graph.crystalgraph import CrystalGraph
import force_benchmark_runner as runner
from cuda_saved_tensor_offload import saved_tensors_on_disk
from relaxation_runner import atomic_json
from validate_inference_optimization import compare


def graphs_for(rows):
    _, packed = runner.prepare_graphs(rows)
    return [CrystalGraph(**{key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                           for key, value in graph.items()}) for graph in packed]


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.9)
    runner.init_graph_worker()
    manifest = json.loads((runner.reproduce.ROOT/'datasets/manifest.json').read_text())
    source = manifest['matpes_pbe']['path']
    rows = list(itertools.islice(runner.structures('matpes_pbe', source, start=201728), 32))
    rows += list(itertools.islice(runner.structures('matpes_pbe', source, start=201762), 1))
    rows += list(itertools.islice(runner.structures('matpes_pbe', source, start=214677), 1))
    graphs = graphs_for(rows)
    model = CHGNet.from_dict(torch.load(runner.WEIGHTS, map_location='cpu', weights_only=False)['model']).eval().to('cuda')
    checks = []
    for deterministic in [False, True]:
        torch.use_deterministic_algorithms(deterministic)
        for size in [1, 16]:
            # Repeatability is checked separately from the storage comparison.
            baseline = runner.predict(model, graphs, size, cleanup_cuda=True)
            repeat = runner.predict(model, graphs, size, cleanup_cuda=True)
            with saved_tensors_on_disk(runner.reproduce.ROOT/'tmp/cuda_offload') as statistics:
                offloaded = runner.predict(model, graphs, size, cleanup_cuda=True)
            difference = compare(baseline, offloaded)
            repeat_difference = compare(baseline, repeat)
            check = {'deterministic': deterministic, 'batch_size': size,
                     'baseline_repeat': repeat_difference, 'offload_compare': difference,
                     'offload_statistics': dict(statistics)}
            checks.append(check)
            print(json.dumps({'event': 'fixture_compare', **check}), flush=True)
            assert all(v['finite'] for v in difference.values())
            assert all(v['finite'] for v in repeat_difference.values())
            if deterministic:
                assert all(v['equal'] for v in difference.values())
                assert all(v['equal'] for v in repeat_difference.values())
            else:
                # CUDA atomic reductions already vary between identical original
                # calls. Record that variation, and prove storage equivalence in
                # the separately repeatable deterministic execution below.
                limits = {'e': 2e-5, 'f': 1e-3, 's': 1e-2, 'm': 1e-4, 'crystal_fea': 1e-3}
                assert all(v['max_abs_difference'] <= limits[k] for k, v in difference.items())
                assert all(v['max_abs_difference'] <= limits[k] for k, v in repeat_difference.items())
    torch.use_deterministic_algorithms(False)
    dense_rows = list(itertools.islice(runner.structures('matpes_pbe', source, start=289780), 1))
    dense_graph = graphs_for(dense_rows)[0]
    oom = False
    try:
        model.predict_graph([dense_graph], task='efsm', return_crystal_feas=True, batch_size=1)
    except torch.cuda.OutOfMemoryError:
        oom = True
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory = {'peak_rss_bytes': 0, 'minimum_available_ram_bytes': psutil.virtual_memory().available}
    stopped = threading.Event()
    def monitor():
        process = psutil.Process()
        while not stopped.wait(0.25):
            memory['peak_rss_bytes'] = max(memory['peak_rss_bytes'], process.memory_info().rss)
            memory['minimum_available_ram_bytes'] = min(memory['minimum_available_ram_bytes'], psutil.virtual_memory().available)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    begun = time.perf_counter()
    try:
        with saved_tensors_on_disk(runner.reproduce.ROOT/'tmp/cuda_offload') as statistics:
            dense_result = model.predict_graph([dense_graph], task='efsm', return_crystal_feas=True, batch_size=1)
        dense_stats = dict(statistics)
    finally:
        stopped.set()
        watcher.join()
    assert all(np.isfinite(np.asarray(v)).all() for v in dense_result.values())
    assert np.asarray(dense_result['f']).shape == (216, 3)
    report = {'validation_pass': True, 'weights_sha256': runner.reproduce.sha(runner.WEIGHTS),
              'implementation_sha256': runner.reproduce.sha(runner.reproduce.ROOT/'cuda_saved_tensor_offload.py'),
              'dtype': 'float32', 'device': 'cuda', 'task': 'efsm', 'tf32': False,
              'memory_fraction': 0.9, 'fixture_indices': [r['index'] for r in rows],
              'checks': checks, 'dense_case': {'index': 289780, 'source_id': dense_rows[0]['source_id'],
              'atoms': 216, 'original_singleton_oom': oom, 'all_fields_finite': True,
              'force_shape': [216, 3], 'seconds': time.perf_counter()-begun,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), **memory,
              'offload_statistics': dense_stats},
              'scope': 'Bitwise exact e/f/s/m/crystal_fea comparisons with deterministic CUDA, at batch sizes 1/16. Non-deterministic comparisons separately record original-repeat and offload floating-point variation. GPU arithmetic is unchanged; actual singleton OOM input retained.'}
    atomic_json(runner.reproduce.OUT/'chgnet_cuda_disk_offload_validation.json', report)
    np.savez_compressed(runner.reproduce.OUT/'chgnet_dense_offload_289780.npz', **dense_result)
    print(json.dumps({'event': 'validation_complete', 'dense_case': report['dense_case']}), flush=True)


if __name__ == '__main__':
    main()
