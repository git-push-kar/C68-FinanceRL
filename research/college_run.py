"""Fresh-PC driver: diagnose, verify SSD base, smoke, train, select and export.

Only standard-library imports at startup: diagnosis works before installing ML.
Every invocation writes a new run directory and a small return-to-chat ZIP.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from collect_results import collect

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def hardware_report():
    report = {"platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
              "disk_free_gib": shutil.disk_usage(ROOT).free / 2 ** 30}
    command = shutil.which("nvidia-smi")
    if command:
        result = subprocess.run([command, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
                                 text=True, capture_output=True)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    else:
        report["nvidia_smi"] = "not found"
    return report


def find_sources(ssd_root=None, explicit=None):
    from base_assets import candidate_sources
    if explicit:
        return [Path(explicit)]
    roots = [Path(ssd_root)] if ssd_root else []
    if not roots and os.name == "nt":
        roots = [Path(f"{letter}:/") for letter in "DEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{letter}:/").exists()]
    return [p for root in roots for p in candidate_sources(root) if p.is_dir()]


def select_checkpoint(run_dir):
    """Validation-only selection, never final-test selection."""
    records = []
    for file in Path(run_dir).glob("validation_*.json"):
        data = json.loads(file.read_text())
        metric = data["learned"]
        sharpe = metric.get("sharpe_proxy")
        if sharpe is None or not math.isfinite(sharpe):
            continue
        steps = int(file.stem.split("_")[-1])
        package = Path(run_dir) / f"checkpoint_{steps}"
        if (package / "manifest.json").is_file():
            records.append({"steps": steps, "validation_sharpe": sharpe, "package": str(package),
                            "return_total": metric["return_total"], "max_drawdown": metric["max_drawdown"]})
    if not records:
        raise ValueError("No finite validation result with a matching saved checkpoint")
    # Prefer the earlier checkpoint on ties. The rule is fixed before training.
    best = max(records, key=lambda x: (x["validation_sharpe"], -x["steps"]))
    return {"rule": "highest validation Sharpe; earliest checkpoint on ties", "candidates": records, "selected": best}


class Runner:
    def __init__(self, directory, dry_run=False):
        self.directory = directory
        self.dry_run = dry_run
        self.environment = os.environ.copy()
        self.environment.update(PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / "model_cache/huggingface"),
                                TOKENIZERS_PARALLELISM="false", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def execute(self, name, arguments):
        command = [sys.executable, "-u", *[str(x) for x in arguments]]
        write_json(self.directory / "commands" / f"{name}.json", {"argv": command, "cwd": str(ROOT)})
        print(f"\n[{name}] {subprocess.list2cmdline(command)}", flush=True)
        if self.dry_run:
            return
        log = self.directory / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line); stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
            finally:
                process.stdout.close()
        write_json(self.directory / "commands" / f"{name}_result.json",
                   {"exit_code": code, "seconds": time.monotonic() - start})
        if code:
            raise RuntimeError(f"Stage {name} failed with exit code {code}. Log: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["diagnose", "smoke", "train"], default="smoke")
    parser.add_argument("--ssd-root")
    parser.add_argument("--base-source")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--offline-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the commands without installing/loading/training")
    args = parser.parse_args()
    if args.steps < 1 or args.minibatch_size is not None and args.minibatch_size < 1:
        parser.error("steps and minibatch size must be positive")
    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "college_runs" / f"{stamp}_{args.stage}"
    directory.mkdir(parents=True)
    write_json(directory / "hardware.json", hardware_report())
    write_json(directory / "request.json", vars(args))
    runner = Runner(directory, args.dry_run)
    status = {"stage": args.stage, "status": "running", "directory": str(directory), "dry_run": args.dry_run,
              "learning_gate_version": 2}
    exit_code = 0
    try:
        if args.stage == "diagnose":
            print(json.dumps(hardware_report(), indent=2))
            status["status"] = "diagnosed"
        else:
            runner.execute("gpu_check", ["-c", "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else 0})); assert torch.cuda.is_available(), 'Install a compatible NVIDIA driver and CUDA PyTorch build'; assert torch.cuda.is_bf16_supported(), 'This recipe requires a bf16-capable GPU'"])
            if not args.dry_run:
                import torch
                vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                free = torch.cuda.mem_get_info()[0] / 2 ** 30
                if free < 5.5:
                    raise RuntimeError(f"Only {free:.1f} GiB GPU memory free. Close other GPU jobs before this full-base run.")
            else:
                vram = 24
            minibatch = args.minibatch_size or (16 if vram >= 20 else 4 if vram >= 12 else 1)
            runner.execute("unit_tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
            base = ROOT / "local_models/InternVL3-2B"
            sources = find_sources(args.ssd_root, args.base_source)
            prepare = ["base_assets.py", "--destination", str(base)]
            for source in sources:
                prepare.extend(["--source", str(source)])
            if args.offline_model:
                prepare.append("--offline")
            runner.execute("prepare_pristine_base", prepare)
            runner.environment["FINANCERL_BASE_MODEL"] = str(base)
            dtype = "bf16" if args.dtype == "auto" else args.dtype
            try:
                runner.execute(f"likelihood_{dtype}", ["likelihood_check.py", "--device", "cuda", "--dtype", dtype,
                                                       "--output", str(directory / f"likelihood_{dtype}.json")])
            except RuntimeError:
                if args.dtype != "auto" or vram < 20 or dtype != "bf16":
                    raise
                print("BF16 likelihood gate failed. Trying full FP32 on this 24GB-class GPU; KL protection stays enabled.", flush=True)
                dtype = "fp32"
                runner.execute("likelihood_fp32", ["likelihood_check.py", "--device", "cuda", "--dtype", dtype,
                                                   "--output", str(directory / "likelihood_fp32.json")])
            status["training_dtype"] = dtype
            runner.execute("pretrained_model_check", ["smoke_backbone.py", "--device", "cuda", "--dtype", dtype,
                                                      "--output", str(directory / "pretrained_check.json")])
            runner.execute("baseline", ["evaluate_baselines.py", "--data", "data/nifty50_prices.csv", "--start", "2023-01-01", "--end", "2024-12-31"])
            for seed in args.seeds:
                steps = 512 if args.stage == "smoke" else args.steps
                training = directory / f"seed_{seed}"
                interval = 256 if args.stage == "smoke" else min(2500, steps)
                runner.execute(f"train_seed_{seed}", ["train.py", "--data", "data/nifty50_prices.csv", "--train-end", "2022-12-31",
                    "--timesteps", steps, "--output", str(training), "--device", "cuda", "--dtype", dtype, "--seed", seed,
                    "--rollout-steps", "128" if args.stage == "smoke" else "256", "--minibatch-size", minibatch,
                    "--checkpoint-interval", interval, "--validation-interval", interval,
                    "--validation-start", "2023-01-01", "--validation-end", "2024-12-31"])
                if args.dry_run:
                    package = training / "final"
                else:
                    learning = json.loads((training / "learning_check.json").read_text())
                    if not learning.get("learning_verified") or learning.get("optimizer_updates", 0) < 1:
                        raise RuntimeError("Training produced no verified LoRA updates; integration cannot pass")
                    selection = select_checkpoint(training)
                    write_json(training / "selection.json", selection)
                    package = Path(selection["selected"]["package"])
                runner.execute(f"reload_and_evaluate_seed_{seed}", ["evaluate_policy.py", "--data", "data/nifty50_prices.csv",
                    "--artifacts", str(package), "--start", "2023-01-01", "--end", "2024-12-31", "--device", "cuda",
                    "--output", str(training / "selected_validation.json")])
                runner.execute(f"export_seed_{seed}", ["export_adapter.py", "--package", str(package),
                    "--output", str(directory / "packages" / f"finance_{args.stage}_seed{seed}.zip")])
            status.update(status="dry_run_plan" if args.dry_run else "passed",
                          interpretation="integration smoke only" if args.stage == "smoke" else "trained validation-selected candidate; not a final-test performance claim")
    except KeyboardInterrupt:
        status.update(status="interrupted", error="Stopped by user")
        (directory / "error.txt").write_text("Run interrupted by user; completed checkpoints remain in the run directory.", encoding="utf-8")
        exit_code = 130
    except Exception as error:
        status.update(status="failed", error=str(error))
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        exit_code = 1
    finally:
        write_json(directory / "status.json", status)
        support = collect(ROOT)
        print(f"\nRESULT DIRECTORY: {directory}\nSEND BACK THIS ZIP: {support}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
