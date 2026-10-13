"""Compare all raw outputs of original and optimized inference on fixed inputs."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import itertools
import json
import time
import numpy as np
import torch
from chgnet.model.model import CHGNet
from chgnet.graph.crystalgraph import CrystalGraph
from pymatgen.core import Structure
import force_benchmark_runner as runner
from inference_optimization import enable, disable
from relaxation_runner import atomic_json


def predict(model, graphs, optimized, device, mode):
    metadata = enable(model) if optimized and mode=='derivative_graph' else (disable(model) or None)
    if device=='cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    begun=time.perf_counter()
    if device=='cpu':
        result=model.predict_graph(graphs,task='efsm',return_crystal_feas=True,batch_size=1)
    else:
        # Production NEB is single-image inference; force shards use batch 16.
        result=[]
        for graph in graphs[:10]:
            result.append(model.predict_graph([graph],task='efsm',return_crystal_feas=True,batch_size=1))
        force_graphs=graphs[10:]
        force_predictions=(runner.predict(model,force_graphs,16) if optimized else
            model.predict_graph(force_graphs,task='efsm',return_crystal_feas=True,batch_size=16))
        result.extend(force_predictions)
    if device=='cuda':
        torch.cuda.synchronize()
    return result, {'seconds':time.perf_counter()-begun,'optimized':optimized,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
        'optimization_metadata':metadata}


def compare(a,b):
    assert len(a)==len(b)
    fields={}
    for left,right in zip(a,b):
        assert left.keys()==right.keys()
        for field in left:
            x,y=np.asarray(left[field]),np.asarray(right[field])
            assert x.shape==y.shape
            row=fields.setdefault(field,{'equal':True,'max_abs_difference':0.0,'finite':True})
            row['equal'] &= bool(np.array_equal(x,y))
            row['finite'] &= bool(np.isfinite(x).all() and np.isfinite(y).all())
            row['max_abs_difference']=max(row['max_abs_difference'],float(np.max(np.abs(x-y))))
    return fields


def validate(device, force_start, deterministic=False, mode='derivative_graph'):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    runner.init_graph_worker()
    reference=runner.reproduce.load(runner.reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')['pathways']
    keys=sorted(reference,key=lambda k:len(reference[k]['full_fp_neb_input']['source_start_endpoint_structure']['sites']))
    structures=[]
    fixtures=[]
    for key in [keys[0],keys[len(keys)//2],keys[-1]]:
        structures.append(Structure.from_dict(reference[key]['full_fp_neb_input']['source_start_endpoint_structure']))
        fixtures.append({'source':'NEB initial','id':key,'atoms':len(structures[-1])})
    checkpoint_path=runner.reproduce.OUT/'neb_full_cpu/jobs/113925_14/runtime_checkpoint.json'
    checkpoint=json.loads(checkpoint_path.read_text())
    for i,s in enumerate(checkpoint['recoverable_images']):
        structures.append(Structure.from_dict(s))
        fixtures.append({'source':'NEB checkpoint','id':'113925|14','image':i,'step':checkpoint['optimizer_step'],'atoms':len(structures[-1])})
    manifest=json.loads((runner.reproduce.ROOT/'datasets/manifest.json').read_text())
    batch=list(itertools.islice(runner.structures('matpes_pbe',manifest['matpes_pbe']['path']),force_start,force_start+32))
    for row in batch:
        structures.append(Structure.from_dict(row['structure']))
        fixtures.append({'source':'MatPES','index':row['index'],'id':row['source_id'],'atoms':len(structures[-1])})
    graphs=[runner.CONVERTER(s) for s in structures]
    model=CHGNet.from_dict(torch.load(runner.WEIGHTS,map_location='cpu',weights_only=False)['model']).eval().to(device)
    predictions={}
    timing=[]
    for label,optimized in [('original',False),('original_repeat',False),('optimized',True)]:
        values,seconds=predict(model,graphs,optimized,device,mode)
        predictions[label]=values
        timing.append({'label':label,**seconds})
        print(json.dumps(timing[-1]),flush=True)
    normal_repeat=compare(predictions['original'],predictions['original_repeat'])
    optimized_compare=compare(predictions['original_repeat'],predictions['optimized'])
    result={'device':device,'optimization_mode':mode,'deterministic_validation':deterministic,'fixtures':fixtures,'structures':len(structures),'task':'efsm','batch_size':1 if device=='cpu' else {'NEB':1,'force':16},
        'dtype':'float32','tf32':False,'weights_sha256':runner.reproduce.sha(runner.WEIGHTS),
        'timing':timing,'repeat_comparison':normal_repeat,'optimized_comparison':optimized_compare,
        'all_outputs_exactly_equal':all(r['equal'] and r['finite'] for r in optimized_compare.values())}
    suffix=('_retry_only' if mode=='retry_only' else '')+('_deterministic' if deterministic else '')
    atomic_json(runner.reproduce.OUT/f'inference_optimization_validation_{device}{suffix}.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--force-start',type=int,default=57344)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--mode',choices=['derivative_graph','retry_only'],default='derivative_graph')
    args=parser.parse_args()
    validate(args.device,args.force_start,args.deterministic,args.mode)
