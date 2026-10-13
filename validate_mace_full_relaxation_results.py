"""Audit all relaxed records, their export, and original public metric tables."""
import datetime
import importlib.metadata
import json
import audit_research_results as audit
import new_potential_relaxation as runner
from relaxation_runner import atomic_json
from threadpoolctl import threadpool_limits


def main():
    jobs,provenance = runner.planned()
    assert len(jobs) == 6697
    folder = runner.OUT
    summary = audit.read(folder/'summary.json')
    assert summary['completed'] == summary['planned'] == 6697
    assert summary['success'] == 6697 and summary['non_converged'] == summary['failed'] == 0
    assert summary['provenance'] == provenance
    with threadpool_limits(limits=1):
        raw = audit.audit_relax('additional',jobs)
    saved = audit.read(folder/'standardized_results.json')['models'][runner.mace.KEY]
    reference,_,_ = audit.reproduce.data()
    ref = reference['reference_data']
    for system,candidates in ref['hull'].items():
        for cid,row in candidates.items():
            key = 'hull||'+(system if row['role']=='interior' else '__endpoint__')+'||'+cid
            assert saved['hull']['relax'][system][cid] == audit.read(folder/'records'/(audit.hashlib.sha256(key.encode()).hexdigest()+'.json'))['result']
    for group,row in ref['ordering'].items():
        for cid in row['orderings']:
            key = 'ordering||'+group+'||'+cid
            assert saved['ordering']['relax'][group][cid] == audit.read(folder/'records'/(audit.hashlib.sha256(key.encode()).hexdigest()+'.json'))['result']
    import convexhull_analysis_utils as utils
    with threadpool_limits(limits=1):
        built = utils.build_phase_stability_ordering_results(ref,{runner.mace.KEY:saved})
        label = {runner.mace.KEY:'MACE-OMAT-0 Small'}
        hull, _ = utils.build_combined_hull_table(built['dft_hull'],built['fp_hull'],[runner.mace.KEY],label)
        ordering, _ = utils.build_combined_ordering_table(built['dft_ordering'],built['fp_ordering'],[runner.mace.KEY],label)
        rmsd, _, _ = utils.build_rmsd_table(built['dft_hull'],built['fp_hull'],[runner.mace.KEY],label)
    table_hashes = {}
    for name,table in [('hull_metrics',hull),('ordering_metrics',ordering),('rmsd_metrics',rmsd)]:
        path = folder/(name+'.csv')
        assert path.read_text(encoding='utf-8').replace('\r\n','\n') == table.to_csv().replace('\r\n','\n'),name
        table_hashes[name] = audit.reproduce.sha(path)
    report = {'pass':True,'scope':'Full additional-model phase stability and ordering relaxation benchmark; force/NEB completion is assessed separately',
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'auditor_sha256':audit.AUDITOR_SHA256,'validator_sha256':audit.reproduce.sha(audit.ROOT/'validate_mace_full_relaxation_results.py'),
        'summary_sha256':audit.reproduce.sha(folder/'summary.json'),
        'standardized_export_sha256':audit.reproduce.sha(folder/'standardized_results.json'),
        'raw_records':raw,'every_exported_relaxation_matches_raw_record':True,
        'all_metric_tables_recomputed_with_original_public_functions':True,'table_sha256':table_hashes,
        'analysis_environment_versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','pymatgen','ase','threadpoolctl']},
        'model_calculation_environment_unchanged':True}
    atomic_json(audit.OUT/'mace_full_relaxation_validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
