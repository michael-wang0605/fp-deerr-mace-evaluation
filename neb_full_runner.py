"""Execute FP-DeErr's full MatCalc NEB template with the verified CHGNet checkpoint."""
import argparse
import ast
import contextlib
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import reproduce
import relaxation_runner as relaxation

OUT = reproduce.OUT / "neb_full"
CALC = None
EXECUTION = None


def output_folder(device,run_name=None):
    if run_name:
        import re
        assert re.fullmatch(r'[a-zA-Z0-9_-]+',run_name)
    return reproduce.OUT/(run_name or ('neb_full_cuda' if device=='cuda' else 'neb_full_cpu'))


def template():
    path = reproduce.REPO / "Ion_migration_NEB/generation/fp_neb_generation_and_run.ipynb"
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "FULL_FP_NEB_TEMPLATE =" in source:
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FULL_FP_NEB_TEMPLATE" for t in n.targets))
            return ast.literal_eval(node.value)
    raise RuntimeError("Full NEB template not found")


def record_path(key):
    return OUT / "records" / (key.replace("|", "_") + ".json")


def initialize(device,run_name=None,inference_optimized=False):
    global CALC, OUT, EXECUTION
    OUT = output_folder(device,run_name)
    relaxation.initialize(1)
    CALC = relaxation.CALC
    if device == 'cuda':
        import torch
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        CALC.device = 'cuda:0'
        CALC.model.to(CALC.device)
    EXECUTION={'inference_optimized':inference_optimized}
    if inference_optimized:
        path=reproduce.OUT/('inference_optimization_validation_cuda_deterministic.json'
            if device=='cuda' else 'inference_optimization_validation_cpu.json')
        validation=json.loads(path.read_text())
        assert validation['all_outputs_exactly_equal']
        from inference_optimization import enable
        EXECUTION['optimization']=enable(CALC.model)
        EXECUTION['validation_sha256']=reproduce.sha(path)


def compute(job, provenance):
    key, pathway = job
    work = OUT / "jobs" / key.replace("|", "_")
    work.mkdir(parents=True, exist_ok=True)
    source = pathway["full_fp_neb_input"]
    input_path = work / "input.json"
    output_path = work / "output.json"
    relaxation.atomic_json(input_path, {"identifiers": pathway["identifiers"],
        "start": source["source_start_endpoint_structure"], "end": source["source_end_endpoint_structure"]})
    substitutions = {"__SITE_PKGS__": str(Path(sys.prefix) / "Lib/site-packages"), "__IMPORT_LINES__": "", "__CALC_LINES__": "calc = supplied_calc",
        "__MODEL_PATH__": "None", "__OUTPUT_KEY__": provenance.get('fp_key', 'CHGNET'), "__PATHWAY_KEY__": key, "__NIMAGES__": "5", "__RELAX_FMAX__": "0.002",
        "__RELAX_MAX_STEPS__": "500", "__NEB_FMAX__": "0.05", "__NEB_MAX_STEPS__": "1000", "__AUTOSORT_TOLS__": "[0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]",
        "__INPUT_PATH__": input_path.as_posix(), "__OUTPUT_PATH__": output_path.as_posix(), "__RUNTIME_CHECKPOINT_PATH__": (work / "runtime_checkpoint.json").as_posix(),
        "__REG_KEY__": provenance.get('registry_key', 'chgnet'), "__DISPLAY_NAME__": provenance.get('model', 'CHGNet'), "__ENV_PATH__": Path(sys.prefix).as_posix(),
        "__SOURCE_REFERENCE_FILE__": (reproduce.REPO / "Ion_migration_NEB/data/ion_migration_neb_reference.json.gz").as_posix(),
        "__SOURCE_REFERENCE_SHA256__": provenance["reference_sha256"]}
    script = template()
    for old, new in substitutions.items():
        script = script.replace(old, new)
    # Inserted path is a Python literal on Windows.
    script = script.replace("sys.path.insert(0, '" + substitutions["__SITE_PKGS__"] + "')", "sys.path.insert(0, " + repr(substitutions["__SITE_PKGS__"]) + ")")
    (work / "generated_run.py").write_text(script, encoding="utf-8")
    begun = time.perf_counter()
    namespace = {"supplied_calc": CALC, "__name__": "__fpdeerr_generated__"}
    prefix, separator, tail = script.partition("with open(INPUT_PATH) as f:")
    assert separator
    with (work / "run.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            exec(compile(prefix, str(work / "generated_run.py"), "exec"), namespace)
            # Persistence adaptation only: serialize numpy scalars and retry Windows file-lock races.
            # Endpoint relaxation, interpolation, NEB construction and convergence logic remain upstream.
            namespace["atomic_write_json"] = relaxation.atomic_json
            exec(compile(separator + tail, str(work / "generated_run.py"), "exec"), namespace)
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as error:
            result = {"identifiers": pathway["identifiers"], "final_fp_neb_images": {}, "neb_status": {"calculation_status": "failed", "neb_converged": None, "error": repr(error)}}
    result["seconds"] = time.perf_counter() - begun
    # Generated observer functions refer to their execution namespace and
    # optimizers. Release those cycles after a completed path; keep only the
    # shared calculator and the serialized scientific result.
    namespace.clear()
    gc.collect()
    relaxation.atomic_json(record_path(key), {"id": key, "provenance": provenance, "result": result,
                                             "execution":EXECUTION})
    return {"id": key, "seconds": result["seconds"], **result["neb_status"]}


def run(args):
    global OUT
    OUT = output_folder(args.device,args.run_name)
    reference_path = reproduce.REPO / "Ion_migration_NEB/data/ion_migration_neb_reference.json.gz"
    reference = reproduce.load(reference_path)
    version_names = ["matcalc", "ase", "pymatgen", "chgnet", "torch", "numpy", "phonopy", "phono3py"]
    provenance = {"reference_sha256": reproduce.sha(reference_path), "template_sha256": hashlib.sha256(template().encode()).hexdigest(),
        "weights_sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1", "device": args.device, "threads": 1,
        "graph_algorithm": "legacy", "atom_cutoff": 6, "bond_cutoff": 3,
        "versions": {name: importlib.metadata.version(name) for name in version_names}}
    jobs = list(reference["pathways"].items())
    if args.pathway:
        jobs = [(key, rec) for key, rec in jobs if key in args.pathway]
        assert len(jobs) == len(args.pathway)
    if args.limit:
        jobs = jobs[:args.limit]
    pending = []
    completed = {}
    for key, rec in jobs:
        path = record_path(key)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["id"] == key and payload["provenance"] == provenance
            if payload["result"]["neb_status"]["calculation_status"] != "failed" or not args.retry_failed:
                completed[key] = payload["result"]
                continue
        pending.append((key, rec))
    print(f"FULL NEB: {len(jobs)} planned, {len(completed)} cached, {len(pending)} pending", flush=True)
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.device,args.run_name,args.inference_optimized)) as pool:
        futures = [pool.submit(compute, job, provenance) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            completed[row["id"]] = json.loads(record_path(row["id"]).read_text(encoding="utf-8"))["result"]
            progress = {"completed": len(completed), "planned": len(jobs), "converged": sum(r["neb_status"]["neb_converged"] is True for r in completed.values()),
                        "failed": sum(r["neb_status"]["calculation_status"] == "failed" for r in completed.values()), "last": row}
            relaxation.atomic_json(OUT / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    execution_counts={'original':0,'inference_create_graph_false':0}
    for key in completed:
        saved=json.loads(record_path(key).read_text(encoding='utf-8'))
        optimized=saved.get('execution',{}).get('inference_optimized',False)
        execution_counts['inference_create_graph_false' if optimized else 'original']+=1
    execution={'workers':args.workers,'inference_optimized_for_new_paths':args.inference_optimized,
               'path_execution_counts':execution_counts}
    if args.inference_optimized:
        execution['implementation_sha256']=reproduce.sha(reproduce.ROOT/'inference_optimization.py')
        execution['validation_report']='optimized_neb_trajectory_validation.json'
    relaxation.atomic_json(OUT / "run_summary.json", {"provenance": provenance, "execution":execution,
        "planned": len(jobs), "completed": len(completed), "pathways": list(completed)})
    if len(jobs) == 154:
        active = {}
        failed = {}
        for key, result in completed.items():
            if result["neb_status"]["calculation_status"] == "failed":
                failed[key] = {"identifiers": result["identifiers"], "calculation_status": "failed", "error": result["neb_status"]["error"]}
            else:
                active[key] = {"identifiers": result["identifiers"], "final_fp_neb_images": result["final_fp_neb_images"], "neb_status": result["neb_status"], "calculation_metadata": result["calculation_metadata"]}
        static = json.loads((reproduce.OUT / "neb_static/results.json").read_text(encoding="utf-8"))
        static["models"]["CHGNET"]["full_fp_neb"] = {"pathways": active, "unsuccessful_pathways": failed}
        static["full_neb_provenance"] = provenance
        relaxation.atomic_json(OUT / "results.json", static)
        print("FULL NEB exported", len(active), "present;", len(failed), "failed", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pathway", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument('--inference-optimized',action='store_true')
    parser.add_argument('--run-name')
    run(parser.parse_args())
