#!/usr/bin/env python3
"""Summarize WPT audit JSON (max ULP / abs error vs tolerance)."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/wpt-trtx-audit.json")
    with path.open() as f:
        r = json.load(f)

    cases = r["cases"]
    flagged = [c for c in cases if c.get("flagged")]
    print(f"Total passed: {r['passed_cases']}")
    print(f"Flagged: {r['flagged_cases']}")
    print()

    by_op: dict[str, list] = defaultdict(list)
    for c in cases:
        by_op[c["operation"]].append(c)

    print("=== Worst max ULP by operation (top 25) ===")
    op_worst = []
    for op, cs in by_op.items():
        ulps = [c.get("maxUlp") or 0 for c in cs if c.get("toleranceKind") == "Ulp"]
        if not ulps:
            continue
        op_worst.append((max(ulps), op, len(cs), sum(1 for c in cs if c.get("flagged"))))
    op_worst.sort(reverse=True)
    for mx, op, n, nf in op_worst[:25]:
        print(f"  {op:30s} max_ulp={mx:8d}  cases={n:4d}  flagged={nf}")

    print()
    print("=== Flagged cases ===")
    for c in sorted(flagged, key=lambda x: -(x.get("maxUlp") or 0)):
        reasons = "; ".join(c.get("flagReasons", []))
        print(
            f"{c['fileName']:40s} | {c['testName'][:55]:55s} | "
            f"kind={c['toleranceKind']} tol={c['toleranceValue']} tight={c.get('tightUlpMinimum')} "
            f"max_ulp={c.get('maxUlp')} max_abs={c.get('maxAbs')} slack={c.get('slackRatio')}"
        )
        print(f"    -> {reasons}")

    print()
    print("=== Tolerance kind breakdown ===")
    kinds: dict[str, int] = defaultdict(int)
    for c in cases:
        kinds[c["toleranceKind"]] += 1
    for k, v in sorted(kinds.items()):
        print(f"  {k}: {v}")

    exact = [c for c in cases if c.get("maxUlp") == 0 and c.get("toleranceKind") == "Ulp"]
    print()
    print(f"=== Exact (max_ulp=0) ULP tests: {len(exact)} / {len(cases)} ===")

    print()
    print("=== Non-zero ULP, unflagged (top 20) ===")
    unflagged_nz = [c for c in cases if not c.get("flagged") and (c.get("maxUlp") or 0) > 0]
    unflagged_nz.sort(key=lambda x: -(x.get("maxUlp") or 0))
    for c in unflagged_nz[:20]:
        print(
            f"  max_ulp={c.get('maxUlp'):5d} tol={c['toleranceValue']:8.0f} "
            f"slack={c.get('slackRatio', 0):.3f} | {c['operation']:20s} | {c['testName'][:50]}"
        )

    print()
    print("=== Rtol tests (conv) ===")
    rtol = [c for c in cases if c.get("toleranceKind") == "Rtol"]
    for c in sorted(rtol, key=lambda x: -(x.get("maxRtol") or 0)):
        print(
            f"  max_rtol={c.get('maxRtol', 0):.6f} tol={c['toleranceValue']} "
            f"max_abs={c.get('maxAbs', 0):.6g} flagged={c.get('flagged')} | {c['testName'][:60]}"
        )

    print()
    print("=== Flag reason categories ===")
    reasons: dict[str, int] = defaultdict(int)
    for c in cases:
        if not c.get("flagged"):
            continue
        for reason in c.get("flagReasons", []):
            if "exceeds tight minimum" in reason:
                reasons["would_fail_tight"] += 1
            elif "wide ULP tolerance" in reason:
                reasons["wide_tolerance_policy"] += 1
            elif "tolerance budget" in reason:
                reasons["high_slack"] += 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    fail_tight = [
        c
        for c in cases
        if c.get("flagged")
        and any("exceeds tight minimum" in x for x in c.get("flagReasons", []))
    ]
    print()
    print(f"=== Would fail at tight minimum ({len(fail_tight)}) ===")
    for c in sorted(fail_tight, key=lambda x: -(x.get("maxUlp") or 0)):
        print(
            f"  max_ulp={c.get('maxUlp')} tol={c['toleranceValue']} "
            f"tight={c['tightUlpMinimum']} max_abs={c.get('maxAbs')} | {c['testName'][:70]}"
        )

    suspicious = [
        c
        for c in cases
        if (c.get("maxUlp") or 0) > c.get("toleranceValue", 0)
        and c.get("toleranceKind") == "Ulp"
    ]
    print()
    print(f"=== max_ulp > tolerance but passed ({len(suspicious)}) ===")
    for c in suspicious:
        print(
            f"  max_ulp={c.get('maxUlp')} tol={c['toleranceValue']} "
            f"max_abs={c.get('maxAbs')} | {c['testName'][:70]}"
        )

    ulps = [c.get("maxUlp") or 0 for c in cases if c.get("toleranceKind") == "Ulp"]
    nz = [u for u in ulps if u > 0]
    print()
    print(f"=== ULP distribution ({len(ulps)} ULP tests) ===")
    print(f"  exact (max_ulp=0): {len(ulps) - len(nz)}")
    print(f"  non-zero: {len(nz)}")
    if nz:
        print(f"  max: {max(nz)}")
        print(f"  median (nonzero): {sorted(nz)[len(nz) // 2]}")

    atol = [c for c in cases if c["toleranceKind"] == "Atol"]
    print()
    print(f"=== Atol tests ({len(atol)}) ===")
    for c in sorted(atol, key=lambda x: -(x.get("maxAbs") or 0))[:10]:
        slack = c.get("slackRatio", 0) or 0
        print(
            f"  max_abs={c.get('maxAbs', 0):.6g} tol={c['toleranceValue']} "
            f"slack={slack:.3f} flagged={c.get('flagged')} | {c['testName'][:55]}"
        )


if __name__ == "__main__":
    main()
