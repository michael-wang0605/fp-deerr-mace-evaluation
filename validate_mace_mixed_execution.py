"""Check actual mixed production records and rejection of altered provenance."""
import copy
import datetime
import json
from pathlib import Path
import numpy as np
import reproduce
from mace_neb_provenance import record_environment,validate_profiles
from relaxation_runner import atomic_json


def main():
    root=reproduce.ROOT
    config_path=root/'outputs/mace_neb_mixed_config.json'
    config=json.loads(config_path.read_text())
    transition=json.loads((root/'outputs/mace_neb_gpu_transition.json').read_text())
    profiles=config['execution_provenances']
    validate_profiles(profiles)
    assert config['gpu_validation_sha256']==reproduce.sha(root/'outputs/mace_cuda_neb_validation.json')
    assert all(reproduce.sha(root/name)==digest for name,digest in transition['scientific_source_sha256'].items())
    folder=root/'outputs/mace_omat_0_small/neb/full'
    before=set(name[:-5].replace('_','|') for name in transition['records_sha256_before'])
    external=set(config['external_active_pathways'])
    eligible=set(config['gpu_eligible_pathways'])
    reference=reproduce.load(root/'FP-DeErr/Ion_migration_NEB/data/ion_migration_neb_reference.json.gz')
    assert not before&external and not before&eligible and not external&eligible
    assert before|external|eligible==set(reference['pathways'])
    assert all(reproduce.sha(folder/'records'/name)==digest for name,digest in transition['records_sha256_before'].items())
    hashes={};devices={};seen=set()
    representative={}
    for path in sorted((folder/'records').glob('*.json')):
        raw=json.loads(path.read_text());key=raw['id']
        assert key in reference['pathways'] and key not in seen
        assert path.name==key.replace('|','_')+'.json'
        seen.add(key)
        device=record_environment(raw,profiles)
        assert raw['provenance']==profiles[device]
        if device=='cuda':
            assert key in eligible
            assert raw['execution']['validation_sha256']==config['gpu_validation_sha256']
            assert raw['execution']['TF32'] is False
        result=raw['result'];meta=result['calculation_metadata']
        assert result['pathway_key']==key and result['identifiers']['pathway_key']==key
        assert result['neb_status']['calculation_status']=='completed'
        expected={'endpoint_fmax_eV_per_angstrom':.002,'endpoint_max_steps':500,
            'neb_fmax_eV_per_angstrom':.05,'neb_max_steps':1000,
            'n_intermediate_images':5,'total_image_count':7,'climbing_image':True,'fixed_cell':True}
        assert all(meta.get(k)==v for k,v in expected.items())
        assert meta['source_reference_sha256']==profiles[device]['reference_sha256']
        images=result['final_fp_neb_images'];assert set(images)=={str(i) for i in range(7)}
        lattice=reference['pathways'][key]['full_fp_neb_input']['source_start_endpoint_structure']['lattice']['matrix']
        for image in images.values():
            forces=np.asarray(image['fp_forces_eV_per_angstrom'])
            assert forces.shape==(len(image['structure']['sites']),3) and np.isfinite(forces).all()
            assert np.isfinite(image['fp_energy_total_eV'])
            assert np.allclose(image['structure']['lattice']['matrix'],lattice,rtol=0,atol=1e-8)
        devices[key]=device;hashes[path.name]=reproduce.sha(path)
        representative.setdefault(device,raw)
    assert set(representative)=={'cpu','cuda'}
    rejected=[]
    for device,raw in representative.items():
        for field,value in [('dtype','float32'),('head','wrong_head'),('cutoff_angstrom',5.),
            ('threads',2),('weights_sha256','changed'),('template_sha256','changed'),
            ('reference_sha256','changed'),('device','cuda' if device=='cpu' else 'cpu')]:
            mutant=copy.deepcopy(raw);mutant['provenance'][field]=value
            try:record_environment(mutant,profiles)
            except AssertionError:rejected.append(device+':'+field)
            else:raise AssertionError('Altered scientific provenance was accepted')
    proof={'pass':True,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'All completed production raw records at this timestamp; full NEB remains required until 154 paths are audited.',
        'completed':len(seen),'planned':154,'execution_counts':{device:list(devices.values()).count(device) for device in profiles},
        'records_sha256':hashes,'config_sha256':reproduce.sha(config_path),
        'validator_sha256':reproduce.sha(Path(__file__)),
        'disjoint_original_external_pending_ownership':True,'original_records_byte_identical':True,
        'unchanged_scientific_source_sha256':transition['scientific_source_sha256'],
        'finite_seven_images_fixed_cells_and_original_thresholds_checked':True,
        'rejected_provenance_mutations':rejected}
    atomic_json(root/'outputs/mace_neb_mixed_execution_validation.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ['records_sha256','unchanged_scientific_source_sha256']}))


if __name__=='__main__':main()
