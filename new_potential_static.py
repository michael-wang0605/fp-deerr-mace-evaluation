"""Evaluate MACE-OMAT-0 Small on the full FP-DeErr static hull/ordering population."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import reproduce
from relaxation_runner import atomic_json

OUT = reproduce.OUT / "mace_omat_0_small"
MODEL = reproduce.ROOT / "models/mace-omat-0-small.model"
MODEL_SHA = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
KEY = "mace_omat_0_small"
CALC = None
ADAPTOR = None


def record_path(key):
    return OUT / "records" / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def initialize(device='cpu'):
    global CALC, ADAPTOR
    import torch
    from mace.calculators import MACECalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    torch.set_num_threads(1)
    assert reproduce.sha(MODEL) == MODEL_SHA
    if device == 'cuda':
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    CALC = MACECalculator(model_paths=str(MODEL), device=device, default_dtype="float64")
    assert CALC.head == "omat_pbe" and CALC.r_max == 6.0
    ADAPTOR = AseAtomsAdaptor()


def compute(job, provenance):
    from pymatgen.core import Structure
    import numpy as np
    key, structure = job
    begun = time.perf_counter()
    try:
        atoms = ADAPTOR.get_atoms(Structure.from_dict(structure))
        atoms.calc = CALC
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        assert np.isfinite(energy) and np.isfinite(forces).all() and forces.shape == (len(atoms), 3)
        result = {"status": "success", "energy_total": energy, "forces": forces.tolist(), "n_atoms": len(atoms)}
    except Exception as error:
        result = {"status": "failed", "error": repr(error)}
    result["seconds"] = time.perf_counter() - begun
    atomic_json(record_path(key), {"id": key, "provenance": provenance, "result": result})
    return key, result


def run(args):
    reference, _, source = reproduce.data()
    ref = reference["reference_data"]
    jobs = {}
    for system, candidates in ref["hull"].items():
        for cid, rec in candidates.items():
            key = "hull||" + (system if rec["role"] == "interior" else "__endpoint__") + "||" + cid
            jobs[key] = rec["relaxed_structure"]
    for group in sorted(ref["ordering"]):
        for cid, rec in ref["ordering"][group]["orderings"].items():
            jobs["ordering||" + group + "||" + cid] = rec["relaxed_structure"]
    assert len(jobs) == 6697
    versions = {p: importlib.metadata.version(p) for p in ["mace-torch", "torch", "torchmetrics", "numpy", "e3nn", "ase", "pymatgen", "matscipy"]}
    provenance = {**source, "model": "MACE-OMAT-0 Small", "weights_sha256": MODEL_SHA, "versions": versions,
                  "protocol": "static on supplied DFT-relaxed geometries", "dtype": "float64", "device": "cpu",
                  "dispersion_correction": False, "threads_per_worker": 1,
                  "model_source": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model"}
    selected = list(jobs.items())[:args.limit] if args.limit else list(jobs.items())
    pending = []
    results = {}
    for key, structure in selected:
        path = record_path(key)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["id"] == key and payload["provenance"] == provenance
            results[key] = payload["result"]
        else:
            pending.append((key, structure))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[name] = "1"
    print(f"MACE STATIC: {len(selected)} planned; {len(results)} cached; {len(pending)} pending", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        futures = [pool.submit(compute, job, provenance) for job in pending]
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
            if len(results) <= 3 or len(results) % 100 == 0 or result["status"] != "success":
                progress = {"completed": len(results), "planned": len(selected), "failed": sum(r["status"] != "success" for r in results.values()),
                            "elapsed_seconds": time.perf_counter() - started, "last_id": key}
                atomic_json(OUT / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    atomic_json(OUT / "run_summary.json", {"provenance": provenance, "completed": len(results), "planned": len(selected),
                "failed": sum(r["status"] != "success" for r in results.values()), "elapsed_seconds": time.perf_counter() - started})
    atomic_json(OUT / "progress.json", {"completed": len(results), "planned": len(selected),
                "failed": sum(r["status"] != "success" for r in results.values()), "complete": True})
    if not args.limit:
        assert len(results) == 6697 and all(r["status"] == "success" for r in results.values())
        analyze(ref, results, provenance)


def analyze(ref, results, provenance):
    import convexhull_analysis_utils as u
    from finalize_report import clean
    fp = {"metadata": provenance, "hull": {"relax": {}, "static": {}}, "ordering": {"relax": {}, "static": {}}}
    for system, candidates in ref["hull"].items():
        fp["hull"]["static"][system] = {cid: results["hull||" + (system if rec["role"] == "interior" else "__endpoint__") + "||" + cid]
                                              for cid, rec in candidates.items()}
    for group, rec in ref["ordering"].items():
        fp["ordering"]["static"][group] = {cid: results["ordering||" + group + "||" + cid] for cid in rec["orderings"]}
    atomic_json(OUT / "standardized_results.json", {"models": {KEY: fp}})
    built = u.build_phase_stability_ordering_results(ref, {KEY: fp})
    hull, hull_details = u.build_combined_hull_table(built["dft_hull"], built["fp_hull"], [KEY], {KEY: "MACE-OMAT-0 Small"})
    ordering, ordering_details = u.build_combined_ordering_table(built["dft_ordering"], built["fp_ordering"], [KEY], {KEY: "MACE-OMAT-0 Small"})
    hull.to_csv(OUT / "hull_metrics.csv")
    ordering.to_csv(OUT / "ordering_metrics.csv")
    values = [str(v) for v in hull.loc["MACE-OMAT-0 Small"].tolist()[4:]]
    values += [str(v) for idx, row in ordering.iterrows() if str(idx[0]).startswith("Static") for v in row.tolist()]
    baseline = json.loads((reproduce.OUT / "fresh_static_summary.json").read_text(encoding="utf-8"))["static_table_comparison"]
    chgnet_values = baseline["actual_hull"] + baseline["actual_ordering"]
    assert len(values) == len(chgnet_values) == 11
    labels = ["Phase stability energy MAE (meV/atom)", "Ground-state agreement (%)", "Within-phase agreement (%)", "Global hull-minimum agreement (%)",
              "Ordering energy MAE (meV/atom)", "Top-1 accuracy (%)", "Recall@3 (%)", "Recall@10 (%)", "Spearman correlation",
              "Ranking-error rate (%)", "Mean / max DFT energy penalty (meV/atom)"]
    summary = {"provenance": provenance, "completed": 6697, "failed": 0, "hull_unique": 597, "ordering_structures": 6100,
               "metrics": [{"metric": label, "chgnet_baseline": a, "mace_omat_0_small": b} for label, a, b in zip(labels, chgnet_values, values)],
               "hull_details": hull_details, "ordering_details": ordering_details}
    (OUT / "summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    text = ["# Additional potential: MACE-OMAT-0 Small", "", "Fresh static calculations completed on the same full population as the CHGNet baseline: 597 unique phase-stability structures and 6,100 ordering structures. All 6,697 calculations succeeded.", "",
            "Used the official small MACE-OMAT-0 checkpoint in a separate environment, float64 on CPU, without added dispersion correction. Predictions use the supplied DFT-relaxed structures. Metrics use the unchanged public FP-DeErr analysis functions and definitions.", "",
            "| Metric | CHGNet baseline | MACE-OMAT-0 Small |", "|---|---:|---:|"]
    text += [f"| {label} | {a} | {b} |" for label, a, b in zip(labels, chgnet_values, values)]
    text += ["", "This compares benchmark performance on this population; it does not establish general model superiority or unseen-data performance. Training-set overlap has not been audited. Full relaxation, force-dataset, and NEB evaluation of this additional potential remain.", "",
             "Model source: [official MACE foundation-model release](" + provenance["model_source"] + "). Checkpoint SHA-256: " + MODEL_SHA + ".", "",
             "Raw predictions are in records/; standardized_results.json preserves this potential as a separate model entry. Package versions and analysis details are in summary.json.", "",
             "```powershell", r".\.venv-mace\Scripts\python.exe -u new_potential_static.py --workers 8", "```", ""]
    (OUT / "report.md").write_text("\n".join(text), encoding="utf-8")
    print(hull.to_string(), flush=True)
    print(ordering.to_string(), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
