"""
run_all.py — orchestrator for the full EDB chain
==========================================================
Runs, in order, everything that produces the paper:

    1. consolidate_data.py    - merge the exports -> EDB_consolidated.csv
    2. completeness.py        - completeness audit (Table I)
    3. pipeline_EDB_v2.py     - ML modelling (Table II, Fig. 5)
    4. regen_pipeline.py      - architecture diagram (Fig. 2)
    5. regen_acquisition.py   - acquisition chain (Fig. 1)
    6. regen_heatmap.py       - calendar heatmap (Fig. 3)
    7. regen_damping.py       - thermal signature (Fig. 4)

Standard usage:
    python run_all.py

Options :
    --skip consolidate         skip a step (keys: consolidate,
                               completeness, pipeline, fig_pipeline,
                               fig_acquisition, fig_heatmap, fig_damping)
    --only pipeline            run ONLY the named step
    --stop-on-error            stop at the first failure
                               (default: carry on with the remaining steps)
    --quiet                    minimal output (no sub-script logs)
    -h, --help                 show this help

Examples:
    python run_all.py                       # run everything
    python run_all.py --skip consolidate    # skip consolidation
    python run_all.py --only pipeline       # modelling step only
    python run_all.py --stop-on-error       # strict stop on failure

Requirements:
    - All scripts must sit in the same directory as run_all.py
    - The source CSVs (sensor_export_1.csv, sensor_export_2.csv) must be
      in that same directory, for step 1
    - The Python dependencies from requirements.txt must be installed

Output:
    An outputs/ directory holding every artefact (CSV, PNG, PDF).
    A final summary printed to the console and saved to
    outputs/run_summary.txt with durations and statuses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Step definition: key, script, label, expected output(s)
STEPS = [
    {
        "key": "consolidate",
        "script": "consolidate_data.py",
        "label": "Data consolidation",
        "expected": ["EDB_consolidated.csv"],
        "in_root": True,   # written to the script directory, not outputs/
    },
    {
        "key": "completeness",
        "script": "completeness.py",
        "label": "Completeness audit (Table I)",
        "expected": [
            "outputs/completeness.csv",
            "outputs/completeness.tex",
        ],
        "in_root": False,
    },
    {
        "key": "pipeline",
        "script": "pipeline_EDB_v2.py",
        "label": "Predictive modelling (Table II, Fig. 5)",
        "expected": [
            "outputs/resultats_v2_openmeteo.csv",
            "outputs/resultats_par_fold.csv",
            "outputs/v2_importance_features.png",
        ],
        "in_root": False,
    },
    {
        "key": "fig_pipeline",
        "script": "regen_pipeline.py",
        "label": "Architecture diagram (Fig. 2)",
        "expected": ["outputs/fig_pipeline.pdf"],
        "in_root": False,
    },
    {
        "key": "fig_acquisition",
        "script": "regen_acquisition.py",
        "label": "Acquisition chain (Fig. 1)",
        "expected": ["outputs/fig_acquisition.pdf"],
        "in_root": False,
    },
    {
        "key": "fig_heatmap",
        "script": "regen_heatmap.py",
        "label": "Calendar heatmap (Fig. 3)",
        "expected": ["outputs/heatmap_calendaire_broken.png"],
        "in_root": False,
    },
    {
        "key": "fig_damping",
        "script": "regen_damping.py",
        "label": "Thermal signature (Fig. 4)",
        "expected": [
            "outputs/amortissement_CEB.png",
            "outputs/peak_lags.csv",
        ],
        "in_root": False,
    },
]


# =========================================================
# CONSOLE PRESENTATION
# =========================================================
class C:
    """ANSI codes. Disabled on Windows when stdout is not a tty."""
    if sys.platform == "win32" and not sys.stdout.isatty():
        BOLD = DIM = RED = GREEN = YELLOW = BLUE = CYAN = RESET = ""
    else:
        BOLD   = "\033[1m"
        DIM    = "\033[2m"
        RED    = "\033[91m"
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        BLUE   = "\033[94m"
        CYAN   = "\033[96m"
        RESET  = "\033[0m"


def banner():
    line = "═" * 67
    print()
    print(f"{C.BOLD}{C.CYAN}╔{line}╗{C.RESET}")
    # Both fields below must be exactly 67 characters wide, the length of
    # the border line, otherwise the box border misaligns.
    print(f"{C.BOLD}{C.CYAN}║{C.RESET}{C.BOLD}{'   ORCHESTRATOR - full EDB chain':<67}{C.CYAN}║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║{C.RESET}{C.DIM}{'   Hygrothermal pipeline - CNRIA 2026 artefact':<67}{C.CYAN}║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚{line}╝{C.RESET}")
    print()


def step_header(idx, total, label):
    bar = "─" * 67
    print(f"{C.CYAN}{bar}{C.RESET}")
    print(f"{C.BOLD}  STEP {idx}/{total}  -  {label}{C.RESET}")
    print(f"{C.CYAN}{bar}{C.RESET}")


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s"


# =========================================================
# RUNNING ONE STEP
# =========================================================
def run_step(step, quiet=False):
    """Run one script and return (success, duration, message)."""
    script_path = SCRIPT_DIR / step["script"]
    if not script_path.exists():
        return False, 0.0, f"Script not found: {step['script']}"

    t0 = time.perf_counter()
    cmd = [sys.executable, str(script_path)]

    try:
        if quiet:
            # Capture stdout/stderr, surface errors only
            result = subprocess.run(
                cmd, cwd=str(SCRIPT_DIR),
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                # Show the last lines to help diagnosis
                tail = "\n".join(result.stderr.strip().splitlines()[-10:])
                return False, time.perf_counter() - t0, tail
        else:
            # Stream straight to the console (normal mode)
            result = subprocess.run(
                cmd, cwd=str(SCRIPT_DIR), check=False,
            )
            if result.returncode != 0:
                return False, time.perf_counter() - t0, \
                    f"Exit code {result.returncode}"

    except FileNotFoundError:
        return False, time.perf_counter() - t0, \
            f"Python not found: {sys.executable}"
    except Exception as e:
        return False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"

    return True, time.perf_counter() - t0, "OK"


def verify_outputs(step):
    """Check that the expected files were created."""
    missing = []
    for f in step["expected"]:
        if not (SCRIPT_DIR / f).exists():
            missing.append(f)
    return missing


# =========================================================
# FINAL REPORT
# =========================================================
def print_summary(results, total_duration):
    line = "═" * 67
    print()
    print(f"{C.BOLD}{C.CYAN}{line}{C.RESET}")
    print(f"{C.BOLD}  SUMMARY{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{line}{C.RESET}")
    print()

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    for r in results:
        if r["status"] == "OK":
            tag = f"{C.GREEN}✓{C.RESET}"
            label_color = C.RESET
        elif r["status"] == "SKIP":
            tag = f"{C.YELLOW}~{C.RESET}"
            label_color = C.DIM
        else:
            tag = f"{C.RED}✗{C.RESET}"
            label_color = C.RED

        duration_str = fmt_duration(r["duration"]) if r["duration"] > 0 else "—"
        print(f"  {tag}  {label_color}{r['label']:<40}{C.RESET}"
              f"  {C.DIM}{duration_str:>10}{C.RESET}")
        if r["status"] == "FAIL" and r["message"]:
            for line_msg in r["message"].split("\n")[:5]:
                print(f"     {C.DIM}{line_msg}{C.RESET}")

    print()
    print(f"  {C.BOLD}Total{C.RESET} : "
          f"{C.GREEN}{n_ok} OK{C.RESET}, "
          f"{C.YELLOW}{n_skip} skipped{C.RESET}, "
          f"{C.RED}{n_fail} failed{C.RESET}")
    print(f"  {C.BOLD}Duration{C.RESET}: {fmt_duration(total_duration)}")

    if (OUTPUT_DIR).exists():
        n_files = sum(1 for _ in OUTPUT_DIR.iterdir() if _.is_file())
        print(f"  {C.BOLD}Outputs{C.RESET}: {n_files} files in "
              f"{C.CYAN}outputs/{C.RESET}")
    print()


def write_summary_file(results, total_duration):
    """Persistent record written to outputs/run_summary.txt."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "run_summary.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
        f.write("EDB orchestrator - run summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date            : {timestamp}\n")
        f.write(f"Dossier         : {SCRIPT_DIR}\n")
        f.write(f"Python          : {sys.executable}\n")
        f.write(f"Total duration  : {fmt_duration(total_duration)}\n\n")

        f.write("Steps:\n")
        for r in results:
            f.write(f"  [{r['status']:<4}] {r['label']:<40} "
                    f"{fmt_duration(r['duration']) if r['duration'] > 0 else '—':>10}\n")
            if r["status"] == "FAIL" and r["message"]:
                for line_msg in r["message"].split("\n")[:10]:
                    f.write(f"           {line_msg}\n")

        if OUTPUT_DIR.exists():
            f.write("\nOutput files:\n")
            for p in sorted(OUTPUT_DIR.iterdir()):
                if p.is_file() and p.name != "run_summary.txt":
                    size_kb = p.stat().st_size / 1024
                    f.write(f"  {p.name:<40}  {size_kb:>8.1f} Ko\n")

    return path


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator for the EDB chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_all.py                       Run everything
  python run_all.py --skip consolidate    Skip consolidation
  python run_all.py --only pipeline       Run the modelling step only
  python run_all.py --stop-on-error       Strict stop on failure
        """,
    )
    parser.add_argument("--skip", action="append", default=[],
                        choices=[s["key"] for s in STEPS],
                        help="Step(s) to skip (repeatable)")
    parser.add_argument("--only", choices=[s["key"] for s in STEPS],
                        help="Run a single step only")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop at the first failure")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (hide sub-script logs)")
    args = parser.parse_args()

    # Decide which steps to run
    if args.only:
        active_keys = {args.only}
    else:
        active_keys = {s["key"] for s in STEPS} - set(args.skip)

    banner()

    # Pre-flight check
    missing_scripts = [s["script"] for s in STEPS
                       if s["key"] in active_keys
                       and not (SCRIPT_DIR / s["script"]).exists()]
    if missing_scripts:
        print(f"{C.RED}{C.BOLD}ERREUR{C.RESET} : scripts introuvables :")
        for s in missing_scripts:
            print(f"  - {s}")
        print(f"\n  Run run_all.py from the directory that holds the chain scripts")
        print(f"  (consolidate_data.py, completeness.py,")
        print(f"  pipeline_EDB_v2.py, regen_*.py).")
        return 1

    print(f"  {C.DIM}Working directory  : {SCRIPT_DIR}{C.RESET}")
    print(f"  {C.DIM}Active steps       : "
          f"{', '.join(s['key'] for s in STEPS if s['key'] in active_keys)}{C.RESET}")
    print()

    # Execution loop
    results = []
    chain_t0 = time.perf_counter()

    for idx, step in enumerate(STEPS, 1):
        if step["key"] not in active_keys:
            results.append({
                "key": step["key"], "label": step["label"],
                "status": "SKIP", "duration": 0.0, "message": "Skipped",
            })
            continue

        step_header(idx, len(STEPS), step["label"])
        print(f"  {C.DIM}Script : {step['script']}{C.RESET}")
        print()

        ok, duration, message = run_step(step, quiet=args.quiet)

        # Post-run verification
        if ok:
            missing = verify_outputs(step)
            if missing:
                ok = False
                message = "Missing output files: " + ", ".join(missing)

        if ok:
            print(f"\n  {C.GREEN}Step completed in {fmt_duration(duration)}{C.RESET}\n")
            results.append({
                "key": step["key"], "label": step["label"],
                "status": "OK", "duration": duration, "message": message,
            })
        else:
            print(f"\n  {C.RED}Failed: {message}{C.RESET}\n")
            results.append({
                "key": step["key"], "label": step["label"],
                "status": "FAIL", "duration": duration, "message": message,
            })
            if args.stop_on_error:
                # Mark the remaining steps as skipped
                for remaining in STEPS[idx:]:
                    if remaining["key"] in active_keys:
                        results.append({
                            "key": remaining["key"], "label": remaining["label"],
                            "status": "SKIP", "duration": 0.0,
                            "message": "Not run, a previous step failed",
                        })
                break

    total_duration = time.perf_counter() - chain_t0

    # Rapport
    print_summary(results, total_duration)
    summary_path = write_summary_file(results, total_duration)
    print(f"  {C.DIM}Summary saved to: {summary_path}{C.RESET}\n")

    # Exit code: 0 if everything passed or was skipped, 1 on any failure
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
