"""Fresh CHGNet static-on-DFT-NEB reproduction; no NEB optimization."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
import reproduce
import relaxation_runner as relaxation

OUT = reproduce.OUT / "neb_static"
OUT.mkdir(exist_ok=True)


def compute(job, provenance):
    from pymatgen.core import Structure
    key, pathway = job
    begun = time.perf_counter()
    images = {}
    errors = {}
    for image in pathway["dft_neb_reference"]["images"]:
        idx = str(image["image_index"])
        try:
            atoms = relaxation.ADAPTOR.get_atoms(Structure.from_dict(image["structure"]))
            atoms.calc = relaxation.CALC
            images[idx] = {"fp_energy_total_eV": float(atoms.get_potential_energy()),
                           "fp_forces_eV_per_angstrom": atoms.get_forces().tolist()}
        except Exception as error:
            errors[idx] = repr(error)
    row = {"identifiers": pathway["identifiers"], "images": images, "image_errors": errors,
           "status": "present" if not errors else "partial", "seconds": time.perf_counter() - begun}
    payload = {"pathway_key": key, "provenance": provenance, "result": row}
    relaxation.atomic_json(OUT / (hashlib.sha256(key.encode()).hexdigest() + ".json"), payload)
    return {"pathway_key": key, "images": len(images), "errors": len(errors), "seconds": row["seconds"]}


def run(args):
    path = reproduce.REPO / "Ion_migration_NEB/data/ion_migration_neb_reference.json.gz"
    assert reproduce.sha(path) == "e211dc4a88fe5739218d6b1e47404624225c5a0d787f4dcce3ef61a5574087c4"
    reference = reproduce.load(path)
    _, base_provenance = relaxation.plan_jobs("hull")
    provenance = {k: base_provenance[k] for k in ["repository_commit", "weights_sha256", "graph_algorithm", "atom_cutoff", "bond_cutoff"]}
    provenance.update(reference_sha256=reproduce.sha(path), protocol="fp_static_on_dft_neb", device="cpu", threads=1)
    jobs = list(reference["pathways"].items())
    if args.limit:
        jobs = jobs[:args.limit]
    completed = {}
    pending = []
    for key, pathway in jobs:
        record_path = OUT / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        if record_path.exists():
            rec = json.loads(record_path.read_text(encoding="utf-8"))
            assert rec["pathway_key"] == key and rec["provenance"] == provenance
            completed[key] = rec["result"]
        else:
            pending.append((key, pathway))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    print(f"NEB STATIC: planned {len(jobs)}, cached {len(completed)}, pending {len(pending)}", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=relaxation.initialize, initargs=(1,)) as pool:
        futures = {pool.submit(compute, job, provenance): job[0] for job in pending}
        for future in as_completed(futures):
            row = future.result()
            key = row["pathway_key"]
            rec = json.loads((OUT / (hashlib.sha256(key.encode()).hexdigest() + ".json")).read_text(encoding="utf-8"))
            completed[key] = rec["result"]
            relaxation.atomic_json(OUT / "progress.json", {"completed_pathways": len(completed), "planned_pathways": len(jobs),
                "completed_images": sum(len(r["images"]) for r in completed.values()), "elapsed_seconds": time.perf_counter() - started, "last": row})
            if len(completed) <= 3 or len(completed) % 10 == 0 or row["errors"]:
                print("NEB STATIC", len(completed), "/", len(jobs), json.dumps(row), flush=True)
    results = {"schema_version": reference["schema_version"], "component": reference["component"], "units": reference["units"],
               "dataset_name": "ion_migration_neb_results",
               "provenance": provenance, "models": {"CHGNET": {"fp_static_on_dft_neb": {"pathways": completed, "unsuccessful_pathways": {}}}}}
    relaxation.atomic_json(OUT / "results.json", results)
    if not args.limit:
        assert len(completed) == 154 and sum(len(r["images"]) for r in completed.values()) == 1078
        assert not any(r["image_errors"] for r in completed.values())
        analyze(reference, results)


def analyze(reference, results):
    import numpy as np
    from finalize_report import clean
    sys.path.insert(0, str(reproduce.REPO / "Ion_migration_NEB/scripts"))
    import neb_analysis as u
    pathways = results["models"]["CHGNET"]["fp_static_on_dft_neb"]["pathways"]
    assert set(pathways) == set(reference["pathways"])
    for key, pathway in reference["pathways"].items():
        assert not pathways[key]["image_errors"]
        for image in pathway["dft_neb_reference"]["images"]:
            prediction = pathways[key]["images"][str(image["image_index"])]
            forces = np.asarray(prediction["fp_forces_eV_per_angstrom"])
            assert np.isfinite(prediction["fp_energy_total_eV"]) and np.isfinite(forces).all()
            assert forces.shape == (len(image["structure"]["sites"]), 3)
    built = u.build_neb_analysis_results(reference, results, expected_pathways=154)
    assert built.validation_report.ok, built.validation_report.summary_df.to_string()
    assert built.coverage_mismatch_df.empty
    protocol = "fp_static_on_dft_neb"
    table = built.barrier_error_summary_by_protocol[protocol]
    table.to_csv(OUT / "barrier_metrics.csv")
    built.protocol_coverage_df.to_csv(OUT / "coverage.csv", index=False)
    built.fp_path_metrics_by_protocol["CHGNET"][protocol].to_csv(OUT / "path_metrics.csv", index=False)
    built.dft_neb_path_force_errors_df.to_csv(OUT / "force_errors.csv", index=False)
    leaderboard = json.loads((reproduce.REPO / "Ion_migration_NEB/data/ion_migration_neb_leaderboard_summary.json").read_text(encoding="utf-8"))
    published = next(row for row in leaderboard["models"] if row["fp_key"] == "CHGNET")
    # Match the upstream leaderboard's pooled forward+backward barrier aggregation.
    row = table.loc["CHGNET"]
    errors = np.concatenate([row["_errors_energy_forward_barrier"], row["_errors_energy_backward_barrier"]])
    actual = {"barrier_mae_static_eV": u.round_half_away_from_zero(float(np.abs(errors).mean()), 4),
              "barrier_rmse_static_eV": u.round_half_away_from_zero(float(np.sqrt(np.mean(errors ** 2))), 4)}
    comparisons = {key: {"published": published[key], "fresh": value, "match": value == published[key]} for key, value in actual.items()}
    summary = {"provenance": results["provenance"], "pathways": 154, "images": 1078,
               "validation_pass": True, "comparisons": comparisons, "pooled_barrier_errors": len(errors)}
    (OUT / "summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# CHGNet static ion-migration reproduction", "", "Fresh CHGNet 0.3.0 calculations completed for all 1,078 supplied DFT-NEB images across 154 pathways. No image calculation failed. The unmodified public analysis validates the input and image correspondence.", "",
             "| Metric | Published (eV) | Fresh (eV) | Match |", "|---|---:|---:|:---:|"]
    lines += [f"| {key} | {c['published']:.4f} | {c['fresh']:.4f} | {'Yes' if c['match'] else 'No'} |" for key, c in comparisons.items()]
    lines += ["", "Barriers use the upstream pooled forward/backward error population and displayed precision. Per-path metrics and force errors are saved alongside this report.", "",
              "This reruns the static-on-DFT-NEB protocol. Full model NEB optimizations and DFT evaluations on model-generated paths remain to be run.", "",
              "```powershell", r".\.venv\Scripts\python.exe -u neb_static_runner.py --workers 4", "```", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args)
