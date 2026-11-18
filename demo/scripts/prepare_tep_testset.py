"""Split the Rieth et al. (2017) TEP test files into small per-run CSVs.

The simulation tab reads one run per file. The Rieth testing files hold
500 runs x 960 samples per fault (millions of rows), so split them once:

    python scripts/prepare_tep_testset.py ^
        --src  D:\\data\\TEP_FaultFree_Testing.csv D:\\data\\TEP_Faulty_Testing.csv ^
        --out  data\\tep_test --runs 1 2

`.RData` sources are read with `pyreadr` if it is installed
(`pip install pyreadr`); otherwise export them to CSV first.
Output files are named `fault{N}_run{R}.csv` and keep the original header
(faultNumber, simulationRun, sample, xmeas_1..xmeas_41, xmv_1..xmv_11).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _split_csv(src: Path, out: Path, runs: set, writers: dict) -> None:
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        low = [h.strip().lower() for h in header]
        f_col, r_col = low.index("faultnumber"), low.index("simulationrun")
        for row in reader:
            run = int(float(row[r_col]))
            if run not in runs:
                continue
            key = (int(float(row[f_col])), run)
            if key not in writers:
                fh_out = open(out / f"fault{key[0]}_run{key[1]}.csv", "w",
                              newline="", encoding="utf-8")
                w = csv.writer(fh_out)
                w.writerow(header)
                writers[key] = (fh_out, w)
                print("writing", fh_out.name, flush=True)
            writers[key][1].writerow(row)


def _split_rdata(src: Path, out: Path, runs: set) -> None:
    import pyreadr  # type: ignore

    for _, df in pyreadr.read_r(str(src)).items():
        df = df[df["simulationRun"].isin(runs)]
        for (fault, run), part in df.groupby(["faultNumber", "simulationRun"]):
            dst = out / f"fault{int(fault)}_run{int(run)}.csv"
            part.to_csv(dst, index=False)
            print("wrote", dst, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default="data/tep_test")
    ap.add_argument("--runs", nargs="+", type=int, default=[1])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = set(args.runs)
    writers: dict = {}
    try:
        for s in args.src:
            src = Path(s)
            if src.suffix.lower() == ".rdata":
                _split_rdata(src, out, runs)
            else:
                _split_csv(src, out, runs, writers)
    finally:
        for fh, _ in writers.values():
            fh.close()
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
