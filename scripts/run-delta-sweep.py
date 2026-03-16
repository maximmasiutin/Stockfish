"""Run instrumented Stockfish at each depth, summarize delta distributions.

Usage:
  python scripts/run-delta-sweep.py --from 1 --to 10
  python scripts/run-delta-sweep.py --exe ./stockfish --to 24 -o results.txt --csv raw.csv

Outputs:
  Console/txt (-o): zero-delta summary table per depth
  CSV (--csv): full delta distribution for every table at every depth
    Columns: depth,table,total_writes,d0,d1,d2,d3,d4_5,d6_9,d10_19,d20_49,d50_99,d100_199,d200p
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

TABLES = ["pawnCorr", "minorCorr", "nonpawnW", "nonpawnB", "contCorr2", "contCorr4"]
BIN_NAMES = ["d0", "d1", "d2", "d3", "d4_5", "d6_9", "d10_19", "d20_49", "d50_99", "d100_199", "d200p"]
DEFAULT_EXE = "stockfish.exe" if sys.platform == "win32" else "./stockfish"


def run_depth(exe, depth, threads=8):
    """Run bench at given depth, return per-table raw rows and d0 summary."""
    cmd = [exe, "bench", "256", str(threads), str(depth)]
    env = os.environ.copy()
    env["BENCH_DEPTH"] = str(depth)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    output = result.stdout + result.stderr

    d0 = {}
    total_writes = 0
    raw_rows = []

    for line in output.splitlines():
        # CSV format: depth,table,total_writes,d0,d1,...,d200p
        parts = line.split(",")
        if len(parts) >= 14:
            try:
                tbl = parts[1]
                if tbl not in TABLES:
                    continue
                tw = int(parts[2])
                bins = [float(x) for x in parts[3:14]]
                d0[tbl] = bins[0]
                total_writes += tw
                row = {"depth": depth, "table": tbl, "total_writes": tw}
                for i, name in enumerate(BIN_NAMES):
                    row[name] = round(bins[i], 4)
                raw_rows.append(row)
                continue
            except (ValueError, IndexError):
                pass

        # Pretty-print fallback (only if no CSV rows found yet)
        if raw_rows:
            continue
        for tbl in TABLES:
            if line.strip().startswith(tbl):
                cells = [c.strip() for c in line.split("|")]
                if len(cells) >= 13:
                    pcts = []
                    for c in cells[2:13]:
                        m = re.search(r"([\d.]+)%", c)
                        pcts.append(float(m.group(1)) if m else 0.0)
                    d0[tbl] = pcts[0]
                    tw_val = 0
                    m_tw = re.search(r"([\d.]+)M", cells[1])
                    if m_tw:
                        tw_val = int(float(m_tw.group(1)) * 1_000_000)
                    else:
                        m_raw = re.search(r"(\d+)", cells[1])
                        if m_raw:
                            tw_val = int(m_raw.group(1))
                    total_writes += tw_val
                    row = {"depth": depth, "table": tbl, "total_writes": tw_val}
                    for i, name in enumerate(BIN_NAMES):
                        row[name] = round(pcts[i], 4) if i < len(pcts) else 0.0
                    raw_rows.append(row)
                break

    return d0, total_writes, raw_rows


def fmt_row(depth, vals, mean, total_writes):
    return ("%5d | %9.1f%% | %10.1f%% | %9.1f%% | "
            "%9.1f%% | %10.1f%% | %10.1f%% | %5.1f%% | "
            "%14s" % (depth, vals[0], vals[1], vals[2],
                      vals[3], vals[4], vals[5], mean,
                      "{:,}".format(total_writes)))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep delta distributions by depth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--exe", default=DEFAULT_EXE,
                        help="Path to stockfish executable (default: %s)" % DEFAULT_EXE)
    parser.add_argument("--from", dest="from_depth", type=int, default=1,
                        help="Starting depth (default: 1)")
    parser.add_argument("--to", type=int, default=8,
                        help="Maximum depth (default: 8)")
    parser.add_argument("-t", "--threads", type=int, default=8,
                        help="Thread count (default: 8)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save summary table to .txt file (flushed per depth)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save full raw delta distribution to CSV file (flushed per depth)")
    args = parser.parse_args()

    header = ("%5s | %10s | %11s | %10s | "
              "%10s | %11s | %11s | %6s | "
              "%14s" % ("Depth", "pawnCorr", "minorCorr", "nonpawnW",
                        "nonpawnB", "contCorr2", "contCorr4", "Mean",
                        "TotalWrites"))
    sep = "-" * len(header)

    outf = None
    if args.output:
        outf = open(args.output, "w", newline="\n")

    csvf = None
    csvw = None
    if args.csv:
        csvf = open(args.csv, "w", newline="")
        fieldnames = ["depth", "table", "total_writes"] + BIN_NAMES
        csvw = csv.DictWriter(csvf, fieldnames=fieldnames)
        csvw.writeheader()
        csvf.flush()

    def emit(line):
        print(line)
        if outf:
            outf.write(line + "\n")
            outf.flush()

    emit(header)
    emit(sep)

    for depth in range(args.from_depth, args.to + 1):
        t0 = time.time()
        print("--- depth %d ..." % depth, end="", flush=True)
        d0, total_writes, raw_rows = run_depth(args.exe, depth, args.threads)
        elapsed = time.time() - t0
        print(" done in %.1fs" % elapsed)

        if not d0:
            row = "%5d | %s" % (depth, "(no data)".center(80))
        else:
            vals = [d0.get(t, 0.0) for t in TABLES]
            mean = sum(vals) / len(vals)
            row = fmt_row(depth, vals, mean, total_writes)

        emit(row)

        if csvw and raw_rows:
            for r in raw_rows:
                csvw.writerow(r)
            csvf.flush()

    if outf:
        outf.close()
        print("\nSaved summary to %s" % args.output)

    if csvf:
        csvf.close()
        print("Saved raw CSV to %s" % args.csv)


if __name__ == "__main__":
    main()
