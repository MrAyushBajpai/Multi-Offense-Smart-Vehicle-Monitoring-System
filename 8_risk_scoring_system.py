"""
risk_scoring.py — Risk-Based Violation Scoring Engine

Calculates a composite risk score for each vehicle by integrating three factors:

  • Frequency  — total number of violations (more = higher risk)
  • Severity   — violation type weight (traffic light > helmet, etc.)
  • Recency    — time decay so recent violations weigh more than old ones

Score formula (per vehicle):
    risk_score = Σ [ severity_weight(v) × recency_decay(v) ]  ×  frequency_multiplier(n)

Risk bands:
    LOW       0  – 24
    MEDIUM   25  – 49
    HIGH     50  – 74
    CRITICAL 75+
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Severity weights per violation type (higher = more dangerous)
SEVERITY_WEIGHTS: dict[str, float] = {
    "red_light":        10.0,   # running a red light — highest danger
    "speeding":          8.0,   # excessive speed
    "wrong_way":         9.0,   # driving against traffic
    "no_helmet":         6.0,   # motorcycle helmet violation
    "no_seatbelt":       5.0,   # seatbelt violation
    "illegal_parking":   2.0,   # minor infraction
    "mobile_phone":      4.0,   # distracted driving
    "unknown":           3.0,   # fallback for unrecognised types
}

# Recency decay: half-life in days — violations older than this count half as much
RECENCY_HALF_LIFE_DAYS: float = 30.0

# Frequency multiplier: log-based so risk grows with repeat offenders but
# doesn't explode for vehicles with dozens of violations
FREQUENCY_BASE: float = math.e   # natural log

# Risk band thresholds
RISK_BANDS: list[tuple[float, str]] = [
    (75.0, "CRITICAL"),
    (50.0, "HIGH"),
    (25.0, "MEDIUM"),
    (0.0,  "LOW"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def severity_weight(violation_type: str) -> float:
    """Return the severity weight for a given violation type."""
    return SEVERITY_WEIGHTS.get(violation_type.lower(), SEVERITY_WEIGHTS["unknown"])


def recency_decay(violation_date: datetime, reference_date: datetime | None = None) -> float:
    """
    Exponential decay based on how many days ago the violation occurred.

    decay = 2^( -days_elapsed / half_life )

    A violation today          → decay = 1.0   (full weight)
    A violation 30 days ago    → decay = 0.5   (half weight)
    A violation 60 days ago    → decay = 0.25  (quarter weight)
    """
    if reference_date is None:
        reference_date = datetime.now()

    days_elapsed = max((reference_date - violation_date).total_seconds() / 86400, 0)
    return math.pow(2, -days_elapsed / RECENCY_HALF_LIFE_DAYS)


def frequency_multiplier(n: int) -> float:
    """
    Log-based multiplier so repeat offenders score higher, but sub-linearly.

    multiplier = 1 + ln(n)   for n >= 1
    """
    return 1.0 + math.log(max(n, 1), FREQUENCY_BASE)


def risk_band(score: float) -> str:
    """Map a numeric score to a human-readable risk band."""
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

def calculate_vehicle_score(
    violations: list[dict],
    reference_date: datetime | None = None,
) -> dict:
    """
    Calculate the risk score for a single vehicle.

    Each violation dict must contain:
        "type"  : str       — violation type key (see SEVERITY_WEIGHTS)
        "date"  : str       — ISO date string "YYYY-MM-DD" or datetime object

    Returns a dict with:
        score, band, frequency, weighted_sum, breakdown
    """
    if not violations:
        return {
            "score":                0.0,
            "band":                 "LOW",
            "frequency":            0,
            "frequency_multiplier": 1.0,
            "weighted_sum":         0.0,
            "breakdown":            [],
        }

    if reference_date is None:
        reference_date = datetime.now()

    breakdown = []
    weighted_sum = 0.0

    for v in violations:
        # Parse date
        date = v["date"] if isinstance(v["date"], datetime) else datetime.fromisoformat(v["date"])
        vtype   = v.get("type", "unknown")
        sw      = severity_weight(vtype)
        rd      = recency_decay(date, reference_date)
        contrib = sw * rd

        weighted_sum += contrib
        breakdown.append({
            "type":             vtype,
            "date":             date.strftime("%Y-%m-%d"),
            "severity_weight":  round(sw, 3),
            "recency_decay":    round(rd, 4),
            "contribution":     round(contrib, 4),
        })

    n     = len(violations)
    fm    = frequency_multiplier(n)
    score = weighted_sum * fm

    return {
        "score":         round(score, 2),
        "band":          risk_band(score),
        "frequency":     n,
        "frequency_multiplier": round(fm, 4),
        "weighted_sum":  round(weighted_sum, 4),
        "breakdown":     sorted(breakdown, key=lambda x: x["contribution"], reverse=True),
    }


def calculate_fleet_scores(
    fleet: dict[str, list[dict]],
    reference_date: datetime | None = None,
) -> list[dict]:
    """
    Score every vehicle in a fleet dict { plate: [violations] }.

    Returns a list of result dicts sorted by score descending (highest risk first).
    """
    results = []
    for plate, violations in fleet.items():
        result = calculate_vehicle_score(violations, reference_date)
        result["plate"] = plate
        result.setdefault("frequency_multiplier", 1.0)
        results.append(result)

    return sorted(results, key=lambda r: r["score"], reverse=True)


def print_report(results: list[dict]) -> None:
    """Print a formatted risk report to stdout."""
    band_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    print("\n" + "═" * 72)
    print("  VEHICLE RISK SCORING REPORT")
    print("═" * 72)
    print(f"  {'Plate':<14} {'Score':>7}  {'Band':<10} {'Violations':>10}  {'Freq ×':>8}")
    print("─" * 72)

    for r in results:
        band_counts[r["band"]] += 1
        band_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["band"], " ")
        print(
            f"  {r['plate']:<14} {r['score']:>7.2f}  "
            f"{band_icon} {r['band']:<8} {r['frequency']:>10}  "
            f"{r['frequency_multiplier']:>8.4f}"
        )

    print("─" * 72)
    print(f"\n  Summary — {len(results)} vehicles scored:")
    for band in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        print(f"    {band:<10} {band_counts[band]:>4}")

    print("═" * 72 + "\n")


def save_results(results: list[dict], filepath: str = "risk_scores.json") -> None:
    """Write the full results list to a JSON file."""
    Path(filepath).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results saved to '{filepath}'")


# ─────────────────────────────────────────────────────────────────────────────
# Run directly → demo with sample fleet data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Fix reference date so the demo output is stable
    REF = datetime(2026, 5, 7)

    def days_ago(n: int) -> str:
        return (REF - timedelta(days=n)).strftime("%Y-%m-%d")

    # Sample fleet: plate → list of violations
    FLEET = {
        "ABC-1234": [
            {"type": "red_light",      "date": days_ago(2)},
            {"type": "red_light",      "date": days_ago(10)},
            {"type": "speeding",       "date": days_ago(5)},
            {"type": "no_seatbelt",    "date": days_ago(20)},
            {"type": "mobile_phone",   "date": days_ago(45)},
        ],
        "XYZ-5678": [
            {"type": "no_helmet",      "date": days_ago(3)},
            {"type": "no_helmet",      "date": days_ago(60)},
        ],
        "DEF-9999": [
            {"type": "wrong_way",      "date": days_ago(1)},
            {"type": "red_light",      "date": days_ago(1)},
            {"type": "speeding",       "date": days_ago(3)},
            {"type": "speeding",       "date": days_ago(7)},
            {"type": "no_seatbelt",    "date": days_ago(12)},
            {"type": "mobile_phone",   "date": days_ago(15)},
            {"type": "red_light",      "date": days_ago(22)},
        ],
        "GHI-4321": [
            {"type": "illegal_parking", "date": days_ago(90)},
        ],
        "JKL-0001": [
            {"type": "speeding",       "date": days_ago(8)},
            {"type": "no_helmet",      "date": days_ago(14)},
            {"type": "red_light",      "date": days_ago(30)},
        ],
        "MNO-7777": [],   # clean record
    }

    print("Running risk_scoring.py — scoring sample fleet...\n")

    results = calculate_fleet_scores(FLEET, reference_date=REF)

    # Console report
    print_report(results)

    # Detailed breakdown for the highest-risk vehicle
    top = results[0]
    print(f"  Detailed breakdown for highest-risk vehicle: {top['plate']}")
    print(f"  Score: {top['score']}  |  Band: {top['band']}\n")
    print(f"  {'Type':<20} {'Date':<12} {'Severity':>9} {'Decay':>8} {'Contrib':>9}")
    print("  " + "─" * 62)
    for b in top["breakdown"]:
        print(
            f"  {b['type']:<20} {b['date']:<12} "
            f"{b['severity_weight']:>9.3f} {b['recency_decay']:>8.4f} {b['contribution']:>9.4f}"
        )

    # Save full results to JSON
    print()
    save_results(results, "risk_scores.json")