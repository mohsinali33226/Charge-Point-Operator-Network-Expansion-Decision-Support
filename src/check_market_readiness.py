r"""Audit market/evidence readiness for a German CPO expansion decision system.

Run from the project root:
    python src\check_market_readiness.py

This is a SAFE validation step: it does NOT change existing site files,
financial assumptions, simulations, scenarios, or investment rankings.

Purpose:
    - Distinguish public retail-price benchmarks from actual CPO margins.
    - Identify missing operator/site-specific input evidence.
    - Mark synthetic simulations and illustrative financial models DEMO_ONLY.
    - Produce actionable data-readiness CSVs for the next finance-engine update.

Optional future format in data/config/cpo_market_evidence_2026.json:
    "site_evidence": {
      "DE_NBG_001": {
        "actual_cpo_net_sales_by_channel_eur_per_kwh": {
          "value": 0.42, "source_type": "OPERATOR_DATA",
          "source": "Internal settlement report ABC", "as_of": "2026-10-01"
        }
      }
    }

NOTE: The example above is a SCHEMA EXAMPLE, not a market price.
A numeric value alone is not verified: source and date are mandatory.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "data" / "config"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

EVIDENCE_FILE = CONFIG_DIR / "cpo_market_evidence_2026.json"
FINANCE_FILE = CONFIG_DIR / "business_assumptions.json"
SITE_FILE = RAW_DIR / "site_master.csv"
DEMAND_FILE = RAW_DIR / "demand_attempts.csv"

STATUS_FILE = PROCESSED_DIR / "cpo_site_readiness.csv"
MISSING_FILE = PROCESSED_DIR / "cpo_missing_market_inputs.csv"
SUMMARY_FILE = PROCESSED_DIR / "cpo_market_audit_summary.json"

# These keys come from the project's existing market-evidence schema.
# A real CPO can mark non-applicable items with documented reasoning in a
# future version, rather than silently setting an arbitrary number.
SOURCE_TYPES = {"OPERATOR_DATA", "ESTIMATE_QUOTE", "LAW_STANDARD", "VENDOR_SPEC"}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing: {path}\nPlace the reference file in the folder shown."
        )
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return result


def load_sites() -> pd.DataFrame:
    if not SITE_FILE.is_file():
        raise FileNotFoundError(f"Missing sites: {SITE_FILE}")
    frame = pd.read_csv(SITE_FILE, dtype={"site_id": str})
    frame.columns = frame.columns.str.strip()
    missing = {"site_id", "city"} - set(frame.columns)
    if missing:
        raise ValueError(f"site_master.csv lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Site master file is empty")
    frame["site_id"] = frame["site_id"].astype(str).str.strip()
    if frame["site_id"].eq("").any() or frame["site_id"].duplicated().any():
        raise ValueError("Site IDs must be nonempty and unique")
    return frame[["site_id", "city"]].copy()


def valid_evidence_entry(entry: object) -> tuple[bool, str]:
    """A value is sourced only if value/type/source/date are all present."""
    if not isinstance(entry, dict):
        return False, "no documented site-specific evidence"
    value = entry.get("value")
    if value is None or value == "":
        return False, "missing value"
    if entry.get("source_type") not in SOURCE_TYPES:
        return False, "missing/invalid source_type"
    source = entry.get("source")
    if not isinstance(source, str) or not source.strip():
        return False, "missing source or quote reference"
    as_of = entry.get("as_of")
    try:
        if not isinstance(as_of, str) or not as_of:
            raise ValueError("no date")
        date.fromisoformat(as_of)
    except ValueError:
        return False, "missing/invalid as_of date (YYYY-MM-DD)"
    # A source URL is optional: many operator records are internal quotes.
    return True, "source documented; still requires manual verification"


def check_public_price_benchmarks(evidence: dict) -> list[str]:
    """Check that public gross retail prices aren't mistaken for CPO margins."""
    issues = []
    items = evidence.get("verified_public_benchmarks", {})
    if not isinstance(items, dict) or not items:
        return ["No public benchmark section found"]
    for key, obj in items.items():
        if not isinstance(obj, dict):
            issues.append(f"{key}: invalid benchmark record")
            continue
        if "end_user_price" in key and obj.get("vat_included") is not True:
            issues.append(f"{key}: VAT basis not explicitly gross")
        if not obj.get("source"):
            issues.append(f"{key}: source URL missing")
        for bound in ("low", "high"):
            if not isinstance(obj.get(bound), (float, int)):
                issues.append(f"{key}: {bound} is missing or not numeric")
        if "low" in obj and "high" in obj:
            if isinstance(obj["low"], (float, int)) and isinstance(obj["high"], (float, int)) and obj["low"] > obj["high"]:
                issues.append(f"{key}: low exceeds high")
    return issues


def main() -> None:
    sites = load_sites()
    evidence = read_json(EVIDENCE_FILE)
    finance = read_json(FINANCE_FILE)
    required_map = evidence.get("site_specific_inputs_required")
    if not isinstance(required_map, dict) or not required_map:
        raise ValueError(
            "Market evidence is missing 'site_specific_inputs_required' "
            "with the required input names."
        )
    by_site = evidence.get("site_evidence", {})
    if not isinstance(by_site, dict):
        raise ValueError("Optional 'site_evidence' must be a JSON object")

    demand_basis = "SYNTHETIC_OR_UNVERIFIED"
    # This project generated synthetic arrivals in demand_attempts.csv.
    # Presence of that CSV does not turn those arrivals into real-world data.
    if not DEMAND_FILE.exists():
        demand_basis = "NO_DEMAND_FILE"

    missing_rows = []
    summary_rows = []
    for site in sites.itertuples(index=False):
        site_id, city = str(site.site_id), str(site.city)
        specific = by_site.get(site_id, {})
        if not isinstance(specific, dict):
            raise ValueError(f"site_evidence['{site_id}'] must be an object")

        documented = 0
        for field in required_map:
            # IMPORTANT: Never use the market's global null placeholders
            # as site-specific evidence and never borrow another site's quote.
            valid, reason = valid_evidence_entry(specific.get(field))
            if valid:
                documented += 1
            else:
                missing_rows.append({
                    "site_id": site_id,
                    "city": city,
                    "required_input": field,
                    "issue": reason,
                    "required_source": "OPERATOR_DATA or dated site-specific quote/spec",
                })

        missing_count = len(required_map) - documented
        # The existing finance model applies illustrative global values even
        # when a site has some documented inputs. Therefore we NEVER mark its
        # output as investment-ready just because all entries are present.
        status = "MISSING_SITE_EVIDENCE" if missing_count else "EVIDENCE_PRESENT_NOT_VALIDATED"
        summary_rows.append({
            "site_id": site_id,
            "city": city,
            "required_inputs": len(required_map),
            "documented_inputs": documented,
            "missing_inputs": missing_count,
            "market_input_status": status,
            "demand_basis": demand_basis,
            "financial_model_basis": "ILLUSTRATIVE_GLOBAL_ASSUMPTIONS",
            "investment_readiness": "DEMO_ONLY",
            "decision_allowed": False,
        })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ready = pd.DataFrame(summary_rows)
    missing = pd.DataFrame(missing_rows, columns=[
        "site_id", "city", "required_input", "issue", "required_source"
    ])
    ready.to_csv(STATUS_FILE, index=False)
    missing.to_csv(MISSING_FILE, index=False)

    benchmark_issues = check_public_price_benchmarks(evidence)
    report = {
        "reference_date": evidence.get("research_cutoff_date", "UNKNOWN"),
        "sites_checked": int(len(ready)),
        "sites_investment_ready": 0,
        "financial_status": "DEMO_ONLY",
        "reason": (
            "Synthetic/unverified demand; missing operator-specific data; "
            "existing financial calculations use illustrative assumptions."
        ),
        "financial_assumption_fields_present": sorted(finance.keys()),
        "public_price_benchmark_issues": benchmark_issues,
        "readiness_csv": str(STATUS_FILE),
        "missing_input_csv": str(MISSING_FILE),
        "caution": (
            "Public gross charging prices must not be substituted for CPO net "
            "sales or contribution margin. No investment decision is authorized."
        ),
    }
    SUMMARY_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n--- GERMAN CPO MARKET / FINANCIAL READINESS ---")
    print(f"Sites checked: {len(ready)} | Required input types per site: {len(required_map)}")
    print(ready[["city", "documented_inputs", "missing_inputs", "investment_readiness"]].to_string(index=False))
    print("\n--- RESULT ---")
    print("Existing business-case NPVs are DEMONSTRATION RESULTS, not invest/do-not-invest advice.")
    print("Gross public charging prices are NOT CPO contribution margins.")
    print(f"Missing documented site inputs: {len(missing)} entries")
    if benchmark_issues:
        print(f"Public benchmark warnings: {len(benchmark_issues)}")
        for issue in benchmark_issues:
            print(f" - {issue}")
    print("\nSaved:")
    print(STATUS_FILE)
    print(MISSING_FILE)
    print(SUMMARY_FILE)
    print("\nNEXT: Replace the demo financial assumptions with source-tracked, site-specific inputs.")


if __name__ == "__main__":
    main()
