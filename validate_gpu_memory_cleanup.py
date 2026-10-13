"""Verify allocator controls preserve CHGNet math on identical force batches."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import gc
import argparse
import itertools
import json
import numpy as np
import torch
import force_benchmark_runner as runner
import reproduce
from chgnet.model.model import CHGNet
from chgnet.graph.crystalgraph import CrystalGraph
from relaxation_runner import atomic_json
from validate_inference_optimization import compare


def main(memory_fraction=0.75):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.init_graph_worker()
    manifest = json.loads((reproduce.ROOT/'datasets/manifest.json').read_text())
    rows = list(itertools.islice(runner.structures('matpes_pbe',manifest['matpes_pbe']['path'],start=201728), 32))
    _, packed = runner.prepare_graphs(rows)
    graphs = [CrystalGraph(**{key:torch.from_numpy(value) if isinstance(value,np.ndarray) else value for key,value in graph.items()}) for graph in packed]
    model = CHGNet.from_dict(torch.load(runner.WEIGHTS,map_location='cpu',weights_only=False)['model']).eval().to('cuda')
    first = runner.predict(model, graphs, 16)
    reserved_before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    reserved_after_cleanup = torch.cuda.memory_reserved()
    second = runner.predict(model, graphs, 16, cleanup_cuda=True)
    difference = compare(first, second)
    assert all(row['equal'] and row['finite'] for row in difference.values())
    report = {'validation_pass': True, 'structures': len(rows), 'dtype':'float32','device':'cuda',
        'memory_fraction':memory_fraction,'prediction_batch_size':16,'weights_sha256':reproduce.sha(runner.WEIGHTS),
        'all_fields_exact':difference,'reserved_before_bytes':reserved_before,'reserved_after_cleanup_bytes':reserved_after_cleanup,
        'scope':'Allocator cap, garbage collection and unused CUDA cache release only. OOM retries retain original existing mini-batch fallback.'}
    filename = 'gpu_memory_cleanup_validation.json' if memory_fraction == 0.75 else f'gpu_memory_cleanup_validation_{memory_fraction:g}.json'
    atomic_json(reproduce.OUT/filename,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-fraction', type=float, default=0.75)
    args = parser.parse_args()
    assert 0 < args.memory_fraction <= 1
    main(args.memory_fraction)
