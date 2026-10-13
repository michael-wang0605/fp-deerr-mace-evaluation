"""Independently audit complete raw populations against the declared research scope."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import datetime
import numpy as np
import reproduce
import relaxation_runner as baseline
import new_potential_static as mace
from relaxation_runner import atomic_json

ROOT = reproduce.ROOT
OUT = reproduce.OUT
AUDITOR_SHA256 = reproduce.sha(Path(__file__))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_prediction(result, atoms):
    assert result['status'] in ['success','non_converged']
    forces = np.asarray(result['forces'])
    assert forces.shape==(atoms,3) and np.isfinite(forces).all()
    assert np.isfinite(result['energy_total'])


def audit_static(model, jobs):
    if model=='baseline':
        records = read(OUT/'chgnet_fresh_static_exact_checkpoint.json')['records']
    else:
        records = {}
        expected = read(mace.OUT/'run_summary.json')['provenance']
        for key,source,_ in jobs:
            path = mace.record_path(key)
            saved = read(path)
            assert saved['id']==key and saved['provenance']==expected
            records[key]=saved['result']
    assert set(records)=={job[0] for job in jobs} and len(records)==6697
    for key,source,_ in jobs:
        result = records[key]
        finite_prediction(result,len(source['relaxed_structure']['sites']))
        assert result['status']=='success'
    return {'complete':True,'structures':6697,'finite_energy_and_force_shapes':True}


def audit_relax(model, jobs):
    from new_potential_relaxation import planned
    folder = OUT/'relaxation' if model=='baseline' else mace.OUT/'relaxation'
    expected = baseline.plan_jobs('all')[1] if model=='baseline' else planned()[1]
    results = {}
    for key,source,_ in jobs:
        path = folder/'records'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        saved = read(path)
        assert saved['id']==key and saved['provenance']==expected
        result=saved['result']
        finite_prediction(result,len(source['initial_structure']['sites']))
        assert np.allclose(result['relaxed_structure']['lattice']['matrix'],source['initial_structure']['lattice']['matrix'],rtol=0,atol=1e-10)
        maximum=float(np.linalg.norm(np.asarray(result['forces']),axis=1).max())
        assert np.isclose(maximum,result['max_force_eV_per_A'],rtol=0,atol=1e-10)
        if result['status']=='success':
            assert maximum<0.01
        results[key]=result['status']
    return {'complete':True,'structures':len(results),'converged':sum(value=='success' for value in results.values()),
        'non_converged':sum(value=='non_converged' for value in results.values()),
        'finite_energy_and_force_shapes':True,'fixed_cells_verified':True,'record_provenance_verified':True}


def audit_neb(model):
    import neb_full_runner
    ref=reproduce.load(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    folder=OUT/'neb_full_cpu' if model=='baseline' else mace.OUT/'neb'
    output=read(folder/'results.json')
    key='CHGNET' if model=='baseline' else 'MACE_OMAT_0_SMALL'
    branch=output['models'][key]
    full=branch['full_fp_neb']
    static=branch['fp_static_on_dft_neb']
    full_folder = folder if model=='baseline' else folder/'full'
    static_folder = OUT/'neb_static' if model=='baseline' else folder/'static'
    full_provenance = read(full_folder/'run_summary.json')['provenance']
    full_run = read(full_folder/'run_summary.json')
    profiles = full_run.get('execution_provenances') if model=='additional' else None
    if profiles:
        from mace_neb_provenance import validate_profiles,record_environment
        assert validate_profiles(profiles)==full_provenance
        gpu_proof=read(OUT/'mace_cuda_neb_validation.json')
        assert gpu_proof['pass'] and len(gpu_proof['cases'])>=2
        assert reproduce.sha(OUT/'mace_cuda_neb_validation.json')==full_run['gpu_validation_sha256']
        assert gpu_proof['provenance']==profiles['cuda']
        assert all(case['pass'] for case in gpu_proof['cases'])
        for case in gpu_proof['cases']:
            source=full_folder/'records'/(case['pathway'].replace('|','_')+'.json')
            assert reproduce.sha(source)==case['cpu_record_sha256']
            assert read(source)['provenance']==profiles['cpu']
        assert set(full_run['path_execution_environment'])==set(ref['pathways'])
    static_provenance = read(static_folder/'results.json')['provenance']
    assert full_provenance['reference_sha256']==reproduce.sha(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    assert full_provenance['template_sha256']==hashlib.sha256(neb_full_runner.template().encode()).hexdigest()
    expected_weights = 'd14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1' if model=='baseline' else mace.MODEL_SHA
    assert full_provenance['weights_sha256']==static_provenance['weights_sha256']==expected_weights
    assert not full['unsuccessful_pathways'] and not static['unsuccessful_pathways']
    assert set(full['pathways'])==set(static['pathways'])==set(ref['pathways'])
    full_images=static_images=converged=0
    for identifier,pathway in ref['pathways'].items():
        record=full['pathways'][identifier]
        raw=read(full_folder/'records'/(identifier.replace('|','_')+'.json'))
        assert raw['id']==identifier
        if profiles:
            assert record_environment(raw,profiles)==full_run['path_execution_environment'][identifier]
        else:
            assert raw['provenance']==full_provenance
        for field in ['identifiers','final_fp_neb_images','neb_status','calculation_metadata']:
            assert record[field]==raw['result'][field]
        static_raw=read(static_folder/(hashlib.sha256(identifier.encode()).hexdigest()+'.json'))
        assert static_raw['pathway_key']==identifier and static_raw['provenance']==static_provenance
        assert static['pathways'][identifier]==static_raw['result']
        assert record['neb_status']['neb_converged'] in [True,False]
        if record['neb_status']['neb_converged']:
            assert record['neb_status']['last_fmax_eV_per_angstrom']<0.05
            converged+=1
        assert set(record['final_fp_neb_images'])=={str(i) for i in range(7)}
        source_cell=pathway['full_fp_neb_input']['source_start_endpoint_structure']['lattice']['matrix']
        for image in record['final_fp_neb_images'].values():
            forces=np.asarray(image['fp_forces_eV_per_angstrom'])
            assert forces.shape==(len(image['structure']['sites']),3) and np.isfinite(forces).all()
            assert np.isfinite(image['fp_energy_total_eV'])
            assert np.allclose(image['structure']['lattice']['matrix'],source_cell,rtol=0,atol=1e-8)
            full_images+=1
        for image in pathway['dft_neb_reference']['images']:
            prediction=static['pathways'][identifier]['images'][str(image['image_index'])]
            forces=np.asarray(prediction['fp_forces_eV_per_angstrom'])
            assert forces.shape==(len(image['structure']['sites']),3) and np.isfinite(forces).all()
            assert np.isfinite(prediction['fp_energy_total_eV'])
            static_images+=1
    assert full_images==static_images==1078
    summary=read(folder/'summary.json')
    assert summary['pathways']==154 and summary['converged']==converged
    metrics_recomputed=False
    if model=='additional':
        from mace_neb_provenance import register_display_name
        register_display_name()
        import neb_analysis as na
        from neb_plots import area_between_curves,simplify_class
        from export_neb_leaderboard import build_leaderboard
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            fresh=build_leaderboard(reproduce.REPO/'Ion_migration_NEB/data/ion_migration_neb_reference.json.gz',
                folder/'results.json',na,area_between_curves,simplify_class,fp_keys=[key])
        assert fresh['models'][0]==summary['metrics']
        assert read(folder/'leaderboard.json')['models'][0]==summary['metrics']
        metrics_recomputed=True
    return {'complete':True,'pathways':154,'full_images':full_images,'static_images':static_images,
        'converged':converged,'non_converged':154-converged,'failed':0,'fixed_cells_verified':True,
        'additional_neb_all_leaderboard_metrics_recomputed':metrics_recomputed,
        'all_force_shapes_and_values_valid':True,'every_path_raw_record_and_provenance_matches_export':True}


def original_force_records(dataset,manifest):
    """Read source identity/forces without rebuilding unused OMat geometries."""
    if dataset=='matpes_pbe':
        import force_benchmark_runner as sources
        yield from sources.structures(dataset,manifest['path'])
    else:
        from ase_db_backends.aselmdb import LMDBDatabase
        database=LMDBDatabase(manifest['path'],readonly=True)
        for index,row in enumerate(database.select()):
            yield {'index':index,'source_id':str(row.id),'forces':row.forces}


def audit_force(dataset,model,deep_source=False):
    sys.path.insert(0,str(reproduce.REPO/'Force_error/scripts'))
    from force_results import build_force_results
    import force_benchmark_runner as sources
    scope=read(ROOT/'research_scope.json')
    name = ('baseline' if model=='baseline' else 'additional') + ('_matpes_pbe' if dataset=='matpes_pbe' else '_omat24')
    folder=(ROOT/scope['required_evidence'][name]['summary']).parent
    manifest=read(ROOT/'datasets/manifest.json')[dataset]
    summary=read(folder/'summary.json')
    provenance=read(folder/'provenance.json')
    assert summary['full_dataset_complete'] and summary['structures']==manifest['structures']
    assert provenance['dataset']==manifest
    assert provenance['weights_sha256']==('d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1' if model=='baseline' else mace.MODEL_SHA)
    assert provenance['dtype']==('float32' if model=='baseline' else 'float64')
    expected=atoms=undefined=cpu_overrides=0
    iterator=iter(original_force_records(dataset,manifest)) if deep_source else None
    fields=['dft_force_magnitude','fp_force_magnitude','force_magnitude_error','force_angle_error','force_vector_error','structure_id']
    metric_arrays={field:[] for field in fields if field!='structure_id'}
    for path in sorted((folder/'shards').glob('*.npz')):
        metadata=read(path.with_suffix('.json'))
        assert metadata['first']==expected and metadata['sha256']==reproduce.sha(path)
        if model=='additional':
            assert metadata['provenance_sha256']==reproduce.sha(folder/'provenance.json')
        with np.load(path,allow_pickle=False) as shard:
            indices=shard['structure_indices']
            offsets=shard['offsets']
            assert np.array_equal(indices,np.arange(metadata['first'],metadata['last']+1))
            assert len(offsets)==len(indices)+1 and offsets[0]==0 and offsets[-1]==metadata['atoms']
            dft,fp=shard['dft_cartesian'],shard['fp_cartesian']
            assert dft.shape==fp.shape==(metadata['atoms'],3) and np.isfinite(dft).all() and np.isfinite(fp).all()
            assert np.isfinite(shard['energy_eV_per_atom']).all()
            dfts=[dft[offsets[i]:offsets[i+1]] for i in range(len(indices))]
            fps=[fp[offsets[i]:offsets[i+1]] for i in range(len(indices))]
            fresh=build_force_results(dfts,{'audit':fps},structure_ids=indices.tolist())['audit']
            for field in fields:
                assert np.array_equal(np.asarray(fresh[field]),shard[field],equal_nan=True),(dataset,model,path.name,field)
                if field in metric_arrays:
                    metric_arrays[field].append(shard[field].copy())
            if deep_source:
                for i,index in enumerate(indices):
                    row=next(iterator)
                    assert row['index']==index and str(row['source_id'])==str(shard['source_ids'][i])
                    assert np.array_equal(np.asarray(row['forces'],dtype=float),dfts[i])
            undefined+=int(np.isnan(shard['force_angle_error']).sum())
        atoms+=metadata['atoms']
        expected=metadata['last']+1
    assert expected==manifest['structures'] and atoms==summary['atoms'] and undefined==summary['undefined_angle_atoms']
    if deep_source:
        assert next(iterator,None) is None
    from force_benchmark_analysis import summary_from_arrays
    arrays={field:np.concatenate(parts) for field,parts in metric_arrays.items()}
    label=summary.get('model','CHGNet' if model=='baseline' else 'MACE-OMAT-0 Small')
    table,_,_,_,finite,standard=summary_from_arrays(dataset,arrays,label)
    assert table.loc[label].to_dict()==summary['summary_metrics']
    assert int(finite.sum())==summary['finite_angle_atoms']
    assert int((standard['dft_force_magnitude']>.01).sum())==summary['atoms_above_DFT_force_0_01']
    return {'complete':True,'structures':expected,'atoms':atoms,'undefined_angles':undefined,
        'all_shard_checksums_and_contiguous_indices_valid':True,'canonical_metrics_recomputed_from_raw_forces':True,
        'all_displayed_force_metrics_recomputed_from_raw_population':True,
        'original_source_ids_and_DFT_forces_checked':deep_source}


def main(deep_source=False):
    scope=read(ROOT/'research_scope.json')
    jobs,_=baseline.plan_jobs('all')
    assert len(jobs)==6697
    checks={}
    for name,spec in scope['required_evidence'].items():
        path=ROOT/spec['summary']
        if not path.exists():
            checks[name]={'complete':False,'reason':'Required full-coverage summary is not yet present'}
            continue
        model='baseline' if name.startswith('baseline') else 'additional'
        try:
            if 'static_phase' in name:
                result=audit_static(model,jobs)
            elif 'neb' in name:
                result=audit_neb(model)
            elif 'matpes' in name or 'omat24' in name:
                dataset='matpes_pbe' if 'matpes' in name else 'omat24'
                result=audit_force(dataset,model,deep_source)
            else:
                selected=jobs if model=='additional' else [job for job in jobs if job[0].startswith('hull||' if 'hull' in name else 'ordering||')]
                result=audit_relax(model,selected)
            result['summary_sha256']=reproduce.sha(path)
            checks[name]=result
        except Exception as error:
            checks[name]={'complete':False,'reason':repr(error)}
    weights={'CHGNet':(ROOT/'.venv/Lib/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar','d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1'),
        'MACE-OMAT-0 Small':(mace.MODEL,mace.MODEL_SHA)}
    for name,(path,expected) in weights.items():
        checks['weights_'+name]={'complete':reproduce.sha(path)==expected,'sha256':expected}
    repo=subprocess.run(['git','-C',str(reproduce.REPO),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    unchanged=subprocess.run(['git','-C',str(reproduce.REPO),'diff','--quiet']).returncode==0
    checks['upstream_source']={'complete':repo==scope['repository_commit'] and unchanged,'commit':repo,'tracked_source_unchanged':unchanged}
    if deep_source:
        for name,row in read(ROOT/'datasets/manifest.json').items():
            checks['dataset_sha_'+name]={'complete':reproduce.sha(Path(row['path']))==row['sha256'],'sha256':row['sha256']}
    report={'pass':all(row['complete'] for row in checks.values()),'deep_source_verification':deep_source,
        'auditor_sha256':AUDITOR_SHA256,
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_sha256':reproduce.sha(ROOT/'research_scope.json'),
        'checks':checks,'optional_VASP_diagnostics_required':False,
        'meaning':'Pass verifies population coverage, data integrity and scientific output invariants; it does not assert exact agreement of every published metric.'}
    atomic_json(OUT/'prof_mo_review/scientific_audit.json',report)
    print(json.dumps({'pass':report['pass'],'complete_checks':sum(row['complete'] for row in checks.values()),'checks':len(checks),
        'incomplete':{name:row['reason'] for name,row in checks.items() if not row['complete'] and 'reason' in row}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--deep-source',action='store_true')
    args=parser.parse_args()
    main(args.deep_source)
