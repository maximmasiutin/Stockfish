"""Measure typical search depth at given time controls.

Usage:
  python scripts/measure-depth-at-tc.py [options]

  Run standard Fishtest TCs (STC, LTC, STC SMP, LTC SMP):
    python scripts/measure-depth-at-tc.py
    python scripts/measure-depth-at-tc.py --exe ./stockfish

  Run custom TC:
    python scripts/measure-depth-at-tc.py --tc 10+0.1 --threads 1
    python scripts/measure-depth-at-tc.py --tc 20+0.2 --threads 8

  Save output:
    python scripts/measure-depth-at-tc.py -o results.txt
    python scripts/measure-depth-at-tc.py -o results.csv
    python scripts/measure-depth-at-tc.py -o results.json

  Output format is auto-detected from file extension (.txt, .csv, .json).

Method:
  Runs Stockfish on 12 middlegame positions using "go movetime", where
  movetime approximates average time per move: (base + 60 * inc) / 60.
  Records final depth and seldepth for each position, computes min, max,
  mean, and median. Trivial endgames (KRK, KBK) are excluded because
  they reach depth 60-80 instantly and skew statistics.
"""

import argparse
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys

DEFAULT_EXE = "stockfish.exe" if sys.platform == "win32" else "./stockfish"

# Middlegame positions from bench set plus additional typical positions.
# Excludes trivial endgames (KRK, KBK) that reach depth 60-80 instantly.
POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10"),
    ("tactical", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("open", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("closed", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("sicilian", "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"),
    ("e4e5", "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"),
    ("ruylopez", "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("qgd", "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"),
    ("middlegame1", "r2qk2r/pppbbppp/2n1pn2/3p4/3P1B2/2NBPN2/PPP2PPP/R2QK2R w KQkq - 4 7"),
    ("middlegame2", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP1BPPP/R2Q1RK1 w - - 0 9"),
    ("complex", "r2q1rk1/1b2bppp/ppnppn2/8/2PNP3/1PN1BP2/P5PP/R2QB1K1 w - - 0 13"),
]

# Standard Fishtest time controls
STANDARD_CONFIGS = [
    {"label": "STC (10+0.1, 1T)",      "threads": 1, "base": 10, "inc": 0.1},
    {"label": "LTC (60+0.6, 1T)",      "threads": 1, "base": 60, "inc": 0.6},
    {"label": "STC SMP (5+0.05, 8T)",  "threads": 8, "base": 5,  "inc": 0.05},
    {"label": "LTC SMP (20+0.2, 8T)",  "threads": 8, "base": 20, "inc": 0.2},
]

AVG_MOVES = 60


def run_config(exe, cfg):
    total_time = cfg["base"] + AVG_MOVES * cfg["inc"]
    movetime_ms = int(total_time / AVG_MOVES * 1000)
    threads = cfg["threads"]

    commands = "uci\n"
    commands += "setoption name Threads value %d\n" % threads
    commands += "setoption name Hash value 256\n"
    commands += "isready\n"

    for name, fen in POSITIONS:
        commands += "position fen %s\n" % fen
        commands += "go movetime %d\n" % movetime_ms

    commands += "quit\n"

    try:
        result = subprocess.run(
            [exe], input=commands, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print("Error: executable not found: %s" % exe, file=sys.stderr)
        return [], [], movetime_ms
    output = result.stdout + result.stderr

    depths = []
    seldepths = []
    last_depth = None
    last_seldepth = None
    for line in output.splitlines():
        m = re.search(r"info depth (\d+) seldepth (\d+) .*score", line)
        if m:
            last_depth = int(m.group(1))
            last_seldepth = int(m.group(2))
        elif line.startswith("bestmove") and last_depth is not None:
            depths.append(last_depth)
            seldepths.append(last_seldepth)
            last_depth = None
            last_seldepth = None

    return depths, seldepths, movetime_ms


def format_results(cfg, depths, seldepths, movetime_ms):
    """Return dict with all results for one config."""
    if not depths:
        return None
    per_position = []
    for i, (d, sd) in enumerate(zip(depths, seldepths)):
        name = POSITIONS[i][0] if i < len(POSITIONS) else str(i + 1)
        per_position.append({"position": name, "depth": d, "seldepth": sd})
    return {
        "label": cfg["label"],
        "threads": cfg["threads"],
        "base": cfg["base"],
        "inc": cfg["inc"],
        "movetime_ms": movetime_ms,
        "positions_tested": len(depths),
        "depth_min": min(depths),
        "depth_max": max(depths),
        "depth_mean": round(statistics.mean(depths), 1),
        "depth_median": round(statistics.median(depths), 1),
        "seldepth_min": min(seldepths),
        "seldepth_max": max(seldepths),
        "seldepth_mean": round(statistics.mean(seldepths), 1),
        "seldepth_median": round(statistics.median(seldepths), 1),
        "per_position": per_position,
    }


def print_results(all_results):
    for r in all_results:
        if r is None:
            continue
        print("=== %s ===" % r["label"])
        print("  movetime: %dms per move" % r["movetime_ms"])
        print("  Positions tested: %d" % r["positions_tested"])
        print()
        print("  %-14s %6s %9s" % ("Position", "Depth", "SelDepth"))
        print("  %-14s %6s %9s" % ("-" * 14, "-" * 6, "-" * 9))
        for p in r["per_position"]:
            print("  %-14s %6d %9d" % (p["position"], p["depth"], p["seldepth"]))
        print()
        print("  Depth:    min=%3d  max=%3d  mean=%5.1f  median=%5.1f" %
              (r["depth_min"], r["depth_max"], r["depth_mean"], r["depth_median"]))
        print("  SelDepth: min=%3d  max=%3d  mean=%5.1f  median=%5.1f" %
              (r["seldepth_min"], r["seldepth_max"], r["seldepth_mean"], r["seldepth_median"]))
        print()

    # Summary table
    print("=== Summary: Median Depth ===")
    print("  %-30s %8s %11s" % ("TC", "Depth", "SelDepth"))
    print("  %-30s %8s %11s" % ("-" * 30, "-" * 8, "-" * 11))
    for r in all_results:
        if r is None:
            continue
        print("  %-30s %8.1f %11.1f" %
              (r["label"], r["depth_median"], r["seldepth_median"]))
    print()


def save_txt(all_results, path):
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_results(all_results)
    sys.stdout = old_stdout
    with open(path, "w", newline="\n") as f:
        f.write(buf.getvalue())


def save_csv(all_results, path):
    rows = []
    for r in all_results:
        if r is None:
            continue
        rows.append({
            "label": r["label"],
            "threads": r["threads"],
            "base": r["base"],
            "inc": r["inc"],
            "movetime_ms": r["movetime_ms"],
            "depth_min": r["depth_min"],
            "depth_max": r["depth_max"],
            "depth_mean": r["depth_mean"],
            "depth_median": r["depth_median"],
            "seldepth_min": r["seldepth_min"],
            "seldepth_max": r["seldepth_max"],
            "seldepth_mean": r["seldepth_mean"],
            "seldepth_median": r["seldepth_median"],
        })
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


def save_json(all_results, path):
    data = [r for r in all_results if r is not None]
    with open(path, "w", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_tc(tc_str):
    """Parse 'base+inc' string, e.g. '10+0.1' -> (10.0, 0.1)."""
    parts = tc_str.split("+")
    if len(parts) != 2:
        raise ValueError("TC must be in format base+inc, e.g. 10+0.1")
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Measure typical search depth at given time controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--exe", default=DEFAULT_EXE,
                        help="Path to stockfish executable (default: %s)" % DEFAULT_EXE)
    parser.add_argument("--tc", type=str, default=None,
                        help="Custom TC as base+inc, e.g. 10+0.1 (overrides standard TCs)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Thread count for custom TC (required with --tc)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save output to file (.txt, .csv, or .json)")
    args = parser.parse_args()

    if args.tc:
        if args.threads is None:
            parser.error("--threads is required when using --tc")
        base, inc = parse_tc(args.tc)
        configs = [{"label": "TC %s, %dT" % (args.tc, args.threads),
                     "threads": args.threads, "base": base, "inc": inc}]
    else:
        configs = STANDARD_CONFIGS

    print("Executable: %s" % args.exe)
    print("Positions: %d middlegame positions" % len(POSITIONS))
    print("Assumed average game length: %d moves" % AVG_MOVES)
    print()

    all_results = []
    for cfg in configs:
        print("Running %s ..." % cfg["label"], flush=True)
        depths, seldepths, movetime_ms = run_config(args.exe, cfg)
        r = format_results(cfg, depths, seldepths, movetime_ms)
        all_results.append(r)
        if r is None:
            print("  NO DATA")
        else:
            print("  median depth=%.1f, seldepth=%.1f" %
                  (r["depth_median"], r["seldepth_median"]))

    print()
    print_results(all_results)

    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext == ".csv":
            save_csv(all_results, args.output)
        elif ext == ".json":
            save_json(all_results, args.output)
        else:
            save_txt(all_results, args.output)
        print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
