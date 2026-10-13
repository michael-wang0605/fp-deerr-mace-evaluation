"""Analyze fresh fixed-cell CHGNet relaxations with unchanged FP-DeErr functions."""
import json
import argparse
import sys
import numpy as np
import pandas as pd
import reproduce
import relaxation_runner as runner
import convexhull_analysis_utils as u
from finalize_report import clean


def analyze():
    jobs, provenance = runner.plan_jobs("hull")
    records = {}
    missing = []
    for key, _, _ in jobs:
        path = runner.record_path(key)
        if not path.exists():
            missing.append(key)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["id"] == key and payload["provenance"] == provenance
        records[key] = payload["result"]
    if missing:
        raise RuntimeError(f"Incomplete relaxation coverage: {len(records)}/{len(jobs)}. Resume relaxation_runner.py.")
    reference, _, _ = reproduce.data()
    ref = reference["reference_data"]
    static = json.loads((reproduce.OUT / "chgnet_fresh_static_exact_checkpoint.json").read_text(encoding="utf-8"))["records"]
    fp = {"metadata": {"protocol": "fresh CHGNet static plus hull relaxation"},
          "hull": {"relax": {}, "static": {}}, "ordering": {"relax": {}, "static": {}}}
    for system, candidates in ref["hull"].items():
        for mode, source in [("relax", records), ("static", static)]:
            fp["hull"][mode][system] = {
                cid: source["hull||" + (system if rec["role"] == "interior" else "__endpoint__") + "||" + cid]
                for cid, rec in candidates.items()}
    for group, row in ref["ordering"].items():
        fp["ordering"]["static"][group] = {cid: static["ordering||" + group + "||" + cid] for cid in row["orderings"]}
    built = u.build_phase_stability_ordering_results(ref, {"chgnet": fp})
    hull, details = u.build_combined_hull_table(built["dft_hull"], built["fp_hull"], ["chgnet"], {"chgnet": "CHGNet"})
    rmsd, rmsd_frames, rmsd_details = u.build_rmsd_table(built["dft_hull"], built["fp_hull"], ["chgnet"], {"chgnet": "CHGNet"})
    hull.to_csv(reproduce.OUT / "chgnet_fresh_relaxation_hull_metrics.csv")
    rmsd.to_csv(reproduce.OUT / "chgnet_fresh_relaxation_rmsd_metrics.csv")
    for key, frame in rmsd_frames.items():
        frame.to_csv(reproduce.OUT / ("chgnet_fresh_relaxation_rmsd_records_" + key + ".csv"), index=False)
    published = json.loads((reproduce.OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    expected = published["comparisons"][0]["expected"][:4] + published["comparisons"][2]["expected"]
    actual = [str(v) for v in hull.loc["CHGNet"].tolist()[:4] + rmsd.loc["CHGNet"].tolist()]
    labels = ["Energy MAE (meV/atom)", "Ground-state agreement (%)", "Within-phase hull-minimum agreement (%)",
              "Global hull-minimum agreement (%)", "Structure mapping success (%)", "Mean / max RMSD (A)",
              "RMSD < 0.05 A (%)", "RMSD < 0.1 A (%)", "RMSD < 0.2 A (%)"]
    comparisons = [{"metric": label, "published": e, "fresh": a, "match": a == e or reproduce.numeric_equal(a, e)}
                   for label, e, a in zip(labels, expected, actual)]
    assert len(comparisons) == 9
    rows = [{"id": key, **{k: rec.get(k) for k in ["status", "n_atoms", "n_steps", "seconds", "energy_total",
            "published_energy_total", "difference_meV_per_atom", "max_force_eV_per_A", "error"]}} for key, rec in records.items()]
    pd.DataFrame(rows).to_csv(reproduce.OUT / "chgnet_fresh_relaxation_energy_comparison.csv", index=False)
    differences = [abs(row["difference_meV_per_atom"]) for row in rows if row["difference_meV_per_atom"] is not None]
    summary = {"provenance": provenance, "planned": len(jobs), "completed": len(records),
               "success": sum(row["status"] == "success" for row in rows),
               "non_converged": sum(row["status"] == "non_converged" for row in rows),
               "failed": sum(row["status"] == "failed" for row in rows), "comparisons": comparisons,
               "matching_metrics": sum(c["match"] for c in comparisons),
               "mean_abs_energy_difference_meV_per_atom": float(np.mean(differences)),
               "max_abs_energy_difference_meV_per_atom": max(differences),
               "within_0_01_meV_per_atom": sum(d <= 0.01 for d in differences),
               "hull_details": details, "rmsd_details": rmsd_details}
    (reproduce.OUT / "fresh_relaxation_summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    text = ["# Fresh CHGNet phase-stability relaxation", "",
            f"Reran all {len(records)} unique phase-stability structures from the supplied initial geometries. "
            f"Converged: {summary['success']}; non-converged: {summary['non_converged']}; failed: {summary['failed']}.", "",
            "Used the exact ENERGY_FUNC from the pinned upstream generation notebook: fixed-cell ASE FIRE, "
            "fmax = 0.01 eV/A, maximum 10,000 steps. CHGNet 0.3.0 used the same verified checkpoint and 6 A / 3 A graph cutoffs as the static reproduction, on CPU with one thread per worker.", "",
            "Metric calculations use the unmodified public benchmark functions and fresh outputs. "
            "The table reports differences directly; a completed calculation is not treated as proof of numerical agreement.", "",
            "| Metric | Published | Fresh | Match at displayed precision |", "|---|---:|---:|:---:|"]
    text += [f"| {c['metric']} | {c['published']} | {c['fresh']} | {'Yes' if c['match'] else 'No'} |" for c in comparisons]
    text += ["", f"Mean / maximum absolute energy difference from supplied relaxed results: {np.mean(differences):.6f} / {max(differences):.6f} meV/atom.",
             f"Structures within the diagnostic 0.01 meV/atom threshold: {summary['within_0_01_meV_per_atom']} / {len(differences)}.", "",
             "Relaxation trajectories can diverge after small floating-point force differences. Any attribution to a particular platform or library remains unverified without the authors' original environment.", "",
             "Raw relaxed structures, energies, forces, step counts, convergence status, and provenance are retained under relaxation/records/. "
             "See [per-structure comparisons](chgnet_fresh_relaxation_energy_comparison.csv) and [machine-readable summary](fresh_relaxation_summary.json).", "",
             "Ordering relaxations, full force datasets, and full NEB optimization remain separate work.", "",
             "```powershell", r".\.venv\Scripts\python.exe -u relaxation_runner.py --scope hull --workers 8 --threads 1",
             r".\.venv\Scripts\python.exe -u relaxation_analysis.py", "```", ""]
    (reproduce.OUT / "relaxation_report.md").write_text("\n".join(text), encoding="utf-8")
    diagnostic_path = reproduce.OUT / "relaxation_outlier_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        same = all(r["energy_total"] == diagnostic["primary_energy_eV"] for r in diagnostic["repeats"])
        with (reproduce.OUT / "relaxation_report.md").open("a", encoding="utf-8") as handle:
            handle.write("\nLargest-energy-discrepancy diagnostic: " + diagnostic["id"] + ". Repeated from the same initial structure with one and two CPU threads; "
                         + ("both repeated energies exactly equal the primary fresh result." if same else "repeated energies differ from the primary result.")
                         + " See [the diagnostic record](relaxation_outlier_diagnostic.json). The discrepancy from the supplied trajectory remains unresolved.\n")
    print(json.dumps({k: summary[k] for k in ["planned", "completed", "success", "non_converged", "failed", "matching_metrics"]}), flush=True)
    print(hull.to_string(), flush=True)
    print(rmsd.to_string(), flush=True)
    return summary


def analyze_ordering():
    jobs, provenance = runner.plan_jobs("all")
    jobs = [j for j in jobs if j[0].startswith("ordering||")]
    assert len(jobs) == 6100
    records = {}
    for key, _, _ in jobs:
        path = runner.record_path(key)
        if not path.exists():
            raise RuntimeError(f"Missing ordering record {key}; finish the full relaxation run first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["id"] == key and payload["provenance"] == provenance
        records[key] = payload["result"]
    reference, _, _ = reproduce.data()
    ref = reference["reference_data"]
    fp = {"metadata": provenance, "hull": {"relax": {}, "static": {}}, "ordering": {"relax": {}, "static": {}}}
    for group, row in ref["ordering"].items():
        fp["ordering"]["relax"][group] = {cid: records["ordering||" + group + "||" + cid] for cid in row["orderings"]}
    built = u.build_phase_stability_ordering_results(ref, {"chgnet": fp})
    table, details = u.build_combined_ordering_table(built["dft_ordering"], built["fp_ordering"], ["chgnet"], {"chgnet": "CHGNet"})
    table.to_csv(reproduce.OUT / "chgnet_fresh_relaxation_ordering_metrics.csv")
    published = json.loads((reproduce.OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    expected = published["comparisons"][1]["expected"][:7]
    actual = [str(v) for idx, row in table.iterrows() if str(idx[0]).startswith("Full") for v in row.tolist()]
    labels = ["Ordering energy MAE (meV/atom)", "Top-1 accuracy (%)", "Recall@3 (%)", "Recall@10 (%)", "Spearman correlation", "Ranking-error rate (%)", "Mean / max DFT energy penalty (meV/atom)"]
    assert len(actual) == len(expected) == 7
    comparisons = [{"metric": label, "published": e, "fresh": a, "match": a == e or reproduce.numeric_equal(a, e)} for label, e, a in zip(labels, expected, actual)]
    rows = [{"id": key, **{k: rec.get(k) for k in ["status", "n_atoms", "n_steps", "seconds", "energy_total", "published_energy_total", "difference_meV_per_atom", "max_force_eV_per_A", "error"]}} for key, rec in records.items()]
    pd.DataFrame(rows).to_csv(reproduce.OUT / "chgnet_fresh_ordering_relaxation_energy_comparison.csv", index=False)
    summary = {"provenance": provenance, "completed": len(records), "planned": 6100, "groups": 305,
               "success": sum(r["status"] == "success" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
               "non_converged": sum(r["status"] == "non_converged" for r in rows), "comparisons": comparisons,
               "matching_metrics": sum(c["match"] for c in comparisons), "ordering_details": details}
    (reproduce.OUT / "fresh_ordering_relaxation_summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Fresh CHGNet ordering relaxations", "", f"Completed all 6,100 candidates across 305 ordering groups. Converged: {summary['success']}; non-converged: {summary['non_converged']}; failed: {summary['failed']}.", "",
             "Used the same verified CHGNet checkpoint and exact upstream fixed-cell FIRE routine as the phase-stability run: fmax 0.01 eV/A, 10,000 maximum steps, initial geometries, CPU with one thread per worker. Metrics use the unchanged public benchmark functions. Non-converged/failed results are retained and handled by those functions.", "",
             "| Metric | Published | Fresh | Match |", "|---|---:|---:|:---:|"]
    lines += [f"| {c['metric']} | {c['published']} | {c['fresh']} | {'Yes' if c['match'] else 'No'} |" for c in comparisons]
    lines += ["", "Raw relaxed structures, energies, forces and convergence records are under relaxation/records/. Per-candidate energy comparisons are in chgnet_fresh_ordering_relaxation_energy_comparison.csv. Exact agreement with supplied trajectories is reported separately from calculation coverage.", "",
              "```powershell", r".\.venv\Scripts\python.exe -u relaxation_runner.py --scope all --workers 16 --threads 1", r".\.venv\Scripts\python.exe -u relaxation_analysis.py --scope ordering", "```", ""]
    (reproduce.OUT / "ordering_relaxation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["completed", "success", "failed", "non_converged", "matching_metrics"]}), flush=True)
    print(table.to_string(), flush=True)
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["hull", "ordering"], default="hull")
    args = parser.parse_args()
    analyze() if args.scope == "hull" else analyze_ordering()
