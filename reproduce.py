from pathlib import Path
import argparse, ast, copy, gzip, hashlib, importlib.metadata, json, os, subprocess, sys, time
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parent
REPO=ROOT/'FP-DeErr'
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(REPO/'Phase_stability_ordering/scripts'))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with gzip.open(p,'rt',encoding='utf-8') as f: return json.load(f)
def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)),encoding='utf-8')
def data():
    rp=REPO/'Phase_stability_ordering/data/phase_stability_ordering_reference.json.gz'
    fp=REPO/'Phase_stability_ordering/data/phase_stability_ordering_results_standardized.json.gz'
    r,f=load(rp),load(fp)
    assert sha(rp)==f['reference']['sha256']
    return r,f,{'reference_sha256':sha(rp),'results_sha256':sha(fp),
       'repository_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()}

class Rows(HTMLParser):
    def __init__(self): super().__init__(); self.rows=[]; self.inbody=False; self.cell=None; self.row=[]
    def handle_starttag(self,tag,attrs):
        if tag=='tbody': self.inbody=True
        if self.inbody and tag=='tr': self.row=[]
        if self.inbody and tag in ('td','th'): self.cell=''
    def handle_data(self,s):
        if self.cell is not None: self.cell+=s
    def handle_endtag(self,tag):
        if self.inbody and tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell=None
        if self.inbody and tag=='tr': self.rows.append(self.row)
        if tag=='tbody': self.inbody=False

def compare_saved(name,df,cell_idx,nb):
    html=''.join(next(o['data']['text/html'] for o in nb['cells'][cell_idx]['outputs'] if 'text/html' in o.get('data',{})))
    parser=Rows(); parser.feed(html)
    expected=[v for row in parser.rows if 'CHGNet' in row for v in row[row.index('CHGNet')+1:]]
    actual_parser=Rows(); actual_parser.feed(df.to_html())
    actual=[v for row in actual_parser.rows if 'CHGNet' in row for v in row[row.index('CHGNet')+1:]]
    ok=[a==b or numeric_equal(a,b) for a,b in zip(actual,expected)]
    result={'name':name,'expected':expected,'actual':actual,'matching_cells':sum(ok),'total_cells':len(expected),'pass':len(actual)==len(expected) and all(ok)}
    print(name,result,flush=True)
    return result
def numeric_equal(a,b):
    try: return float(a)==float(b)
    except ValueError: return False

def analysis():
    import convexhull_analysis_utils as u
    r,f,provenance=data()
    b=u.build_phase_stability_ordering_results(r['reference_data'],{'chgnet':f['models']['chgnet']})
    validation=u.validate_phase_stability_ordering_results(b['dft_hull'],b['fp_hull'],b['dft_ordering'],b['fp_ordering'],['chgnet'])
    names={'chgnet':'CHGNet'}
    hull,hd=u.build_combined_hull_table(b['dft_hull'],b['fp_hull'],['chgnet'],names)
    ordering,od=u.build_combined_ordering_table(b['dft_ordering'],b['fp_ordering'],['chgnet'],names)
    rmsd,rd,rs=u.build_rmsd_table(b['dft_hull'],b['fp_hull'],['chgnet'],names)
    nb=json.loads((REPO/'Phase_stability_ordering/analysis/convexhull_ordering_analysis_all_models.ipynb').read_text(encoding='utf-8'))
    comparisons=[compare_saved(n,df,i,nb) for n,df,i in [('hull',hull,15),('ordering',ordering,20),('rmsd',rmsd,22)]]
    for name,df in [('hull',hull),('ordering',ordering),('rmsd',rmsd)]:
        df.to_csv(OUT/('chgnet_'+name+'_recomputed.csv'))
    for k,v in rd.items(): v.to_csv(OUT/('chgnet_rmsd_records_'+k+'.csv'),index=False)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','pymatgen','ase','chgnet','torch']}
    write(OUT/'analysis_summary.json',{'provenance':provenance,'versions':versions,'tables_match_saved_outputs':all(c['pass'] for c in comparisons),'comparisons':comparisons,'validation':validation,'hull_details':hd,'ordering_details':od,'rmsd_details':rs})
    report=['# FP-DeErr initial reproduction: CHGNet','',
      'Recomputed the full phase-stability and elemental-ordering CHGNet tables from the shipped model outputs using the unmodified public analysis functions.',
      'This is an analysis reproduction. It does not by itself rerun model inference or relaxation.','',
      'Repository commit: '+provenance['repository_commit'],
      'Reference SHA-256: '+provenance['reference_sha256'],
      'Results SHA-256: '+provenance['results_sha256'],'',
      'The reference checksum agrees with the reference recorded by the results file.',
      'Population: 22 tie-line systems, 597 unique hull structures, 305 ordering groups with 20 candidates each.','',
      'Saved notebook table comparison: '+str(sum(c['matching_cells'] for c in comparisons))+'/'+str(sum(c['total_cells'] for c in comparisons))+' cells match.','']
    for name,df in [('Phase stability',hull),('Ordering',ordering),('Structure relaxation error',rmsd)]:
        report += ['## '+name,'','```text',df.to_string(),'```','']
    report += ['## Re-run','','Run from this workspace:','```powershell','.\\.venv\\Scripts\\python.exe -u reproduce.py analysis','```','',
      'Public repository: https://github.com/mogroupumd/FP-DeErr',
      'Published source tables: FP-DeErr/Phase_stability_ordering/analysis/convexhull_ordering_analysis_all_models.ipynb']
    (OUT/'reproduction_report.md').write_text('\n'.join(report),encoding='utf-8')
    assert all(c['pass'] for c in comparisons), 'At least one table differs from the saved published notebook.'
    print('FULL ANALYSIS REPRODUCTION PASSED',flush=True)

def inspect():
    import inspect as ins
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    print(ins.getsource(CHGNet.load)); print(ins.signature(CHGNetCalculator))
    print('PRETRAINED',list((Path(ins.getfile(CHGNet)).parent.parent/'pretrained').rglob('*')))
    r,f,p=data()
    print('MODEL METADATA',json.dumps(f['models']['chgnet']['metadata'],indent=2))
    print('FIRST HULL',next(iter(r['reference_data']['hull'])))
    print('FIRST ORDERING',next(iter(r['reference_data']['ordering'])))

def static(scope):
    import torch, numpy as np
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    import convexhull_analysis_utils as u
    torch.set_num_threads(2)
    r,f,provenance=data(); ref=r['reference_data']; canonical=f['models']['chgnet']
    weights=ROOT/'.venv/Lib/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar'
    assert sha(weights)=='d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1', 'Checkpoint hash mismatch'
    state=torch.load(weights,map_location='cpu',weights_only=False)
    model=CHGNet.from_dict(state['model'])
    print('OFFICIAL CHECKPOINT SHA256',sha(weights),flush=True)
    from chgnet.graph.converter import CrystalGraphConverter
    model.graph_converter=CrystalGraphConverter(atom_graph_cutoff=model.graph_converter.atom_graph_cutoff,bond_graph_cutoff=model.graph_converter.bond_graph_cutoff,algorithm='legacy')
    model.eval()
    calc=CHGNetCalculator(model=model,use_device='cpu')
    adaptor=AseAtomsAdaptor()
    checkpoint=OUT/'chgnet_fresh_static_exact_checkpoint.json'
    saved=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {'provenance':provenance,'records':{}}
    assert saved['provenance']==provenance
    calc_meta={'weights_sha256':sha(weights),'graph_algorithm':'legacy','atom_cutoff':model.graph_converter.atom_graph_cutoff,'bond_cutoff':model.graph_converter.bond_graph_cutoff,'torch':torch.__version__}
    assert saved.get('calculator',calc_meta)==calc_meta, 'Cached predictions use a different calculator configuration.'
    saved['calculator']=calc_meta
    jobs={}
    for system,candidates in ref['hull'].items():
        for cid,rec in candidates.items():
            key='hull||'+(system if rec['role']=='interior' else '__endpoint__')+'||'+cid
            jobs[key]=(rec,canonical['hull']['static'][system][cid])
    groups=sorted(ref['ordering']) if scope=='all' else [sorted(ref['ordering'])[0]]
    for group in groups:
        for cid,rec in ref['ordering'][group]['orderings'].items():
            jobs['ordering||'+group+'||'+cid]=(rec,canonical['ordering']['static'][group][cid])
    start=time.perf_counter()
    for idx,(key,(rec,published)) in enumerate(jobs.items(),1):
        if key in saved['records'] and saved['records'][key]['status']=='success': continue
        atoms=adaptor.get_atoms(Structure.from_dict(rec['relaxed_structure'])); atoms.calc=calc
        begun=time.perf_counter()
        try:
            e=float(atoms.get_potential_energy()); forces=atoms.get_forces().tolist()
            saved['records'][key]={'status':'success','energy_total':e,'forces':forces,'n_atoms':len(atoms),
                'published_energy_total':published.get('energy_total'),'seconds':time.perf_counter()-begun}
        except Exception as ex:
            saved['records'][key]={'status':'failed','error':repr(ex)}
            write(checkpoint,saved)
            raise
        if idx%50==0 or idx==len(jobs): write(checkpoint,saved)
        if idx<=3 or idx%25==0: print(f'STATIC {idx}/{len(jobs)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    write(checkpoint,saved)
    rows=[]
    for key in jobs:
        z=saved['records'][key]
        if z['status']=='success' and z['published_energy_total'] is not None:
            delta=z['energy_total']-z['published_energy_total']
            rows.append({'id':key,'fresh_energy_eV':z['energy_total'],'published_energy_eV':z['published_energy_total'],
                'difference_eV':delta,'difference_meV_per_atom':delta*1000/z['n_atoms'],'n_atoms':z['n_atoms']})
    import pandas as pd
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'chgnet_fresh_static_energy_comparison.csv',index=False)
    fp={'metadata':{'protocol':'fresh static only'},'hull':{'relax':{},'static':{}},'ordering':{'relax':{},'static':{}}}
    for system,candidates in ref['hull'].items():
        fp['hull']['static'][system]={}
        for cid,rec in candidates.items():
            key='hull||'+(system if rec['role']=='interior' else '__endpoint__')+'||'+cid
            fp['hull']['static'][system][cid]=saved['records'][key]
    for group in groups:
        fp['ordering']['static'][group]={cid:saved['records']['ordering||'+group+'||'+cid] for cid in ref['ordering'][group]['orderings']}
    b=u.build_phase_stability_ordering_results(ref,{'chgnet':fp})
    ht,hd=u.build_combined_hull_table(b['dft_hull'],b['fp_hull'],['chgnet'],{'chgnet':'CHGNet'})
    ht.to_csv(OUT/'chgnet_fresh_static_hull_metrics.csv')
    absdelta=frame['difference_meV_per_atom'].abs()
    summary={'provenance':provenance,'scope':scope,'device':'cpu','threads':2,'checkpoint':'CHGNet 0.3.0',
        'planned_structures':len(jobs),'completed':len(rows),'failed':sum(saved['records'][k]['status']!='success' for k in jobs),
        'hull_unique_structures':597,'ordering_groups':groups,'max_abs_difference_meV_per_atom':float(absdelta.max()),
        'mean_abs_difference_meV_per_atom':float(absdelta.mean()),'within_0_01_meV_per_atom':int((absdelta<=0.01).sum()),
        'fresh_hull_details':hd,'elapsed_seconds':time.perf_counter()-start}
    write(OUT/'fresh_static_summary.json',summary)
    print('FRESH STATIC SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['fresh_hull_details','ordering_groups']},default=str),flush=True)
    print(ht.to_string(),flush=True)

    ordering_table,ordering_details=u.build_combined_ordering_table(b['dft_ordering'],b['fp_ordering'],['chgnet'],{'chgnet':'CHGNet'})
    ordering_table.to_csv(OUT/'chgnet_fresh_static_ordering_metrics.csv')
    published_summary=json.loads((OUT/'analysis_summary.json').read_text(encoding='utf-8'))
    fresh_hull=[str(v) for v in ht.loc['CHGNet'].tolist()[4:]]
    expected_hull=published_summary['comparisons'][0]['expected'][4:]
    fresh_ordering=[str(v) for idx,row in ordering_table.iterrows() if str(idx[0]).startswith('Static') for v in row.tolist()]
    expected_ordering=published_summary['comparisons'][1]['expected'][7:]
    summary['calculator']={'weights_sha256':sha(weights),'source_url':'https://raw.githubusercontent.com/CederGroupHub/chgnet/v0.3.0/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar',
       'graph_algorithm':'legacy','atom_graph_cutoff_A':model.graph_converter.atom_graph_cutoff,'bond_graph_cutoff_A':model.graph_converter.bond_graph_cutoff}
    summary['static_table_comparison']={'expected_hull':expected_hull,'actual_hull':fresh_hull,'expected_ordering':expected_ordering,'actual_ordering':fresh_ordering,
       'hull_pass':all(numeric_equal(a,e) or a==e for a,e in zip(fresh_hull,expected_hull)),
       'ordering_pass':scope=='all' and len(fresh_ordering)==len(expected_ordering) and all(numeric_equal(a,e) or a==e for a,e in zip(fresh_ordering,expected_ordering))}
    write(OUT/'fresh_static_summary.json',summary)
    report_path=OUT/'reproduction_report.md'
    base=report_path.read_text(encoding='utf-8').split('## Fresh model calculations')[0]
    lines=['## Fresh model calculations','',
       'Model: CHGNet 0.3.0, official checkpoint SHA-256 '+sha(weights),
       'Fresh static evaluations on DFT-relaxed structures: '+str(len(rows))+' of '+str(len(jobs))+'; failures: '+str(summary['failed'])+'.',
       'Maximum absolute energy difference from shipped model outputs: '+format(summary['max_abs_difference_meV_per_atom'],'.8f')+' meV/atom.',
       'Mean absolute energy difference: '+format(summary['mean_abs_difference_meV_per_atom'],'.8f')+' meV/atom.',
       'Energies within 0.01 meV/atom: '+str(summary['within_0_01_meV_per_atom'])+' of '+str(len(rows))+'.',
       'Static hull metrics match the published displayed values: '+str(summary['static_table_comparison']['hull_pass'])+'.',
       'Static ordering metrics match the published displayed values: '+str(summary['static_table_comparison']['ordering_pass'])+'.','',
       'The Windows CHGNet 0.3.0 wheel omits the pretrained checkpoint. It was downloaded from the official v0.3.0 Git tag.',
       'The Windows compiled graph extension raises a buffer-dtype error. This run uses the built-in legacy Python graph builder, preserving checkpoint cutoffs (6 A atoms, 3 A bonds).',
       'Device: CPU, two PyTorch threads; dtype: float32 from the checkpoint. Energy equality is evaluated separately from agreement with published metric values.','',
       'The earlier incorrect 5 A cutoff diagnostic run is retained in chgnet_fresh_static_checkpoint.json and is excluded from all successful results here.','',
       'Fresh full structural relaxations, force-prediction datasets, and ion-migration calculations have not been reproduced. The relaxation metrics above were recomputed from supplied model outputs.','',
       'Run the completed static workflow:','```powershell','.\\.venv\\Scripts\\python.exe -u reproduce.py static --scope all','```','']
    report_path.write_text(base+'\n'+'\n'.join(lines),encoding='utf-8')
    print('STATIC COMPARISON',summary['static_table_comparison'],flush=True)
    if scope=='all':
        assert summary['completed']==6697 and summary['failed']==0
        # The 0.01 meV/atom threshold is a diagnostic, not an upstream metric acceptance criterion.
        if summary['within_0_01_meV_per_atom']!=6697:
            print('ENERGY DIAGNOSTIC: one or more differences exceed 0.01 meV/atom; inspect the CSV.',flush=True)
        assert summary['static_table_comparison']['hull_pass'] and summary['static_table_comparison']['ordering_pass']


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['analysis','inspect','static'])
    p.add_argument('--scope',choices=['hull','all'],default='hull')
    args=p.parse_args()
    if args.mode=='analysis': analysis()
    elif args.mode=='inspect': inspect()
    else: static(args.scope)




    if (OUT/'fresh_static_summary.json').exists() and args.mode!='inspect':
        from finalize_report import finalize
        finalize()
