# %% K-Sensitivity Analysis Script
# Runs the full pipeline for multiple values of k (number of archetypes)
# and collects metrics for comparison.

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.json"
RESULTS_DIR = ROOT / "results" / "k_sensitivity"


def parse_args():
    parser = argparse.ArgumentParser(description="K-sensitivity analysis for archetype generation")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., cite_seq, tonsil)")
    parser.add_argument("--k_values", type=str, default="7,9,11", help="Comma-separated k values to test (default: 7,9,11)")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip steps 0-2 (assume already run)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max_epochs in config (default: use config value)")
    return parser.parse_args()


def run_step(cmd, step_name):
    """Run a pipeline step via subprocess, streaming output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {step_name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Force non-interactive backend, no GUI windows
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{step_name} completed in {elapsed:.1f}s")
    return elapsed


def backup_config():
    """Backup original config and return its contents."""
    with open(CONFIG_PATH, "r") as f:
        original = json.load(f)
    backup_path = CONFIG_PATH.with_suffix(".json.bak")
    shutil.copy2(CONFIG_PATH, backup_path)
    print(f"Config backed up to {backup_path}")
    return original


def write_speed_config(original_config, max_epochs_override=None):
    """Write speed-optimized config for sensitivity runs."""
    config = json.loads(json.dumps(original_config))  # deep copy
    config["plot_flag"] = False
    config.setdefault("training", {})
    config["training"]["plot_x_times"] = 0
    config["training"]["save_checkpoint_every_n_epochs"] = 999
    if max_epochs_override is not None:
        config["training"]["max_epochs"] = max_epochs_override
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote speed-optimized config (plot_flag=false, plot_x_times=0, save_checkpoint_every_n_epochs=999)")


def restore_config(original_config):
    """Restore original config."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(original_config, f, indent=2)
    backup_path = CONFIG_PATH.with_suffix(".json.bak")
    if backup_path.exists():
        backup_path.unlink()
    print("Config restored to original")


def collect_mlflow_metrics(dataset_name):
    """Collect metrics from the latest MLflow run for the given dataset."""
    try:
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        exp = mlflow.get_experiment_by_name(dataset_name)
        if exp is None:
            print(f"WARNING: No MLflow experiment found for '{dataset_name}'")
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            print("WARNING: No MLflow runs found")
            return {}

        run = runs.iloc[0]
        metric_cols = [c for c in run.index if c.startswith("metrics.")]
        metrics = {}
        for col in metric_cols:
            val = run[col]
            if val is not None and str(val) != "nan":
                metrics[col.replace("metrics.", "")] = val
        metrics["run_id"] = run["run_id"]
        return metrics
    except Exception as e:
        print(f"WARNING: Could not collect MLflow metrics: {e}")
        return {}


def get_preprocessing_script(dataset_name):
    """Find the Step 0 preprocessing script for the dataset."""
    scripts_dir = ROOT / "scripts"
    candidates = list(scripts_dir.glob(f"_0_preprocess_{dataset_name}*.py"))
    if not candidates:
        candidates = list(scripts_dir.glob("_0_preprocess_*.py"))
        if len(candidates) == 1:
            return str(candidates[0])
        print(f"ERROR: Cannot find preprocessing script for '{dataset_name}'")
        print(f"Available: {[c.name for c in candidates]}")
        sys.exit(1)
    return str(candidates[0])


def main():
    args = parse_args()
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    dataset_name = args.dataset_name

    print(f"\n{'#'*60}")
    print(f"# K-SENSITIVITY ANALYSIS")
    print(f"# Dataset: {dataset_name}")
    print(f"# K values: {k_values}")
    print(f"# Skip preprocessing: {args.skip_preprocessing}")
    print(f"{'#'*60}\n")

    python = sys.executable
    original_config = backup_config()
    all_results = {}

    try:
        write_speed_config(original_config, args.max_epochs)

        # Run Steps 0-2 once (shared across all k values)
        if not args.skip_preprocessing:
            preprocess_script = get_preprocessing_script(dataset_name)
            run_step([python, preprocess_script], "Step 0: Preprocessing")
            run_step(
                [python, str(ROOT / "scripts" / "_1_align_datasets.py"), "--dataset_name", dataset_name],
                "Step 1: Align datasets",
            )
            run_step(
                [python, str(ROOT / "scripts" / "_2_spatial_integrate.py"), "--dataset_name", dataset_name],
                "Step 2: Spatial integration",
            )
        else:
            print("Skipping Steps 0-2 (--skip_preprocessing)")

        # Run Steps 3-5 for each k value
        for k in k_values:
            print(f"\n{'*'*60}")
            print(f"* K = {k}")
            print(f"{'*'*60}")

            k_start = time.time()

            run_step(
                [python, str(ROOT / "scripts" / "_3_generate_archetypes.py"), "--dataset_name", dataset_name, "--force_k", str(k)],
                f"Step 3: Generate archetypes (k={k})",
            )
            run_step(
                [python, str(ROOT / "scripts" / "_4_prepare_training.py"), "--dataset_name", dataset_name],
                f"Step 4: Prepare training (k={k})",
            )
            run_step(
                [python, str(ROOT / "scripts" / "_5_train_vae.py"), "--dataset_name", dataset_name],
                f"Step 5: Train VAE (k={k})",
            )

            k_elapsed = time.time() - k_start
            metrics = collect_mlflow_metrics(dataset_name)
            metrics["k_value"] = k
            metrics["runtime_seconds"] = round(k_elapsed, 1)
            all_results[k] = metrics

            print(f"\nk={k} completed in {k_elapsed:.1f}s")
            if metrics:
                for name in ["NMI", "matching_accuracy", "mixing_score_ilisi", "silhouette_f1", "ari_f1"]:
                    if name in metrics:
                        print(f"  {name}: {metrics[name]:.4f}")

    finally:
        restore_config(original_config)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"k_sensitivity_{dataset_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    csv_path = RESULTS_DIR / f"k_sensitivity_{dataset_name}_{timestamp}.csv"
    if all_results:
        all_keys = set()
        for m in all_results.values():
            all_keys.update(m.keys())
        all_keys = sorted(all_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for k in sorted(all_results.keys()):
                writer.writerow(all_results[k])
        print(f"CSV saved to {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header_metrics = ["k_value", "runtime_seconds", "NMI", "matching_accuracy", "mixing_score_ilisi", "silhouette_f1", "ari_f1"]
    available = [m for m in header_metrics if any(m in r for r in all_results.values())]
    header = " | ".join(f"{m:>22}" for m in available)
    print(header)
    print("-" * len(header))
    for k in sorted(all_results.keys()):
        row = all_results[k]
        vals = []
        for m in available:
            v = row.get(m, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:>22.4f}")
            else:
                vals.append(f"{str(v):>22}")
        print(" | ".join(vals))

    print(f"\nDone! Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
