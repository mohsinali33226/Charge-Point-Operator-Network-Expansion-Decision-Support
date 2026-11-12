"""Match observed German charger occupancy stations with the BNetzA register.

Inputs (existing project files):
    data/processed/german_real_station_quality_checks.csv
    data/processed/german_occupancy_candidate_stations.csv (fallback)
    data/external/Ladesaeulenregister_BNetzA*.csv

Outputs:
    data/processed/german_occupancy_infrastructure_matches.csv
    data/processed/german_occupancy_infrastructure_match_summary.csv

This is an infrastructure identification step, NOT a validation of charging
arrivals, rejected drivers, site grid limits, or charger power sharing.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"

CANDIDATE_FILES = [
    PROCESSED / "german_real_station_quality_checks.csv",
    PROCESSED / "german_occupancy_candidate_stations.csv",
]

ESSENTIAL_REGISTRY_COLUMNS = [
    "Ladeeinrichtungs-ID",
    "Anzahl Ladepunkte",
    "Art der Ladeeinrichtung",
    "Nennleistung Ladeeinrichtung [kW]",
]

OTHER_REGISTRY_COLUMNS = [
    "Betreiber",
    "Anzeigename (Karte)",
    "Status",
    "Ort",
    "Bundesland",
    "Postleitzahl",
    "Breitengrad",
    "Längengrad",
    "Inbetriebnahmedatum",
]



def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().strip('"')
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def number(values: pd.Series) -> pd.Series:
    # German register can use decimal commas and dots for thousand separators.
    # These power / count fields normally use decimal commas, not thousands.
    return pd.to_numeric(
        values.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def read_candidates() -> tuple[pd.DataFrame, str]:
    source = next((p for p in CANDIDATE_FILES if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "No processed candidate file found. Run profile_real_german_stations.py "
            "or analyze_german_occupancy.py first."
        )
    df = pd.read_csv(source, dtype={"station_id": "string"}, low_memory=False)
    df.columns = df.columns.str.strip()
    if "station_id" not in df.columns:
        raise ValueError(f"{source.name} has no station_id column.")
    df["station_id"] = df["station_id"].map(normalize_id)
    df = df.loc[df["station_id"] != ""].copy()
    df = df.drop_duplicates(subset=["station_id"], keep="first")
    if df.empty:
        raise ValueError("Candidate file contains no usable station IDs.")
    wanted = ["station_id"] + [
        c for c in (
            "valid_snapshots", "valid_status_observations", "observed_days",
            "typical_total_points", "mean_occupied_pct",
            "no_available_snapshot_pct", "validation_candidate"
        ) if c in df.columns
    ]
    return df[wanted].copy(), source.name


def locate_registry_header(path: Path) -> tuple[str, int, list[str]]:
    # The BNetzA CSV includes multi-line explanatory text before its header.
    # Detect the real header instead of treating the metadata as CSV records.
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                for line_index, line in enumerate(f):
                    if line_index > 45:
                        break
                    cells = next(csv.reader([line], delimiter=";"))
                    cells = [c.strip().lstrip("\ufeff") for c in cells]
                    if cells and cells[0] == "Ladeeinrichtungs-ID":
                        return encoding, line_index, cells
        except UnicodeDecodeError:
            continue
    raise ValueError(
        "Cannot find the BNetzA data header ('Ladeeinrichtungs-ID') "
        "in the first 46 lines. Check the downloaded CSV."
    )


def read_registry_matches(
    path: Path, candidates: set[str], chunk_size: int
) -> tuple[pd.DataFrame, int, list[str]]:
    encoding, header_index, header = locate_registry_header(path)
    missing = [name for name in ESSENTIAL_REGISTRY_COLUMNS if name not in header]
    if missing:
        raise ValueError(
            f"BNetzA header is missing: {missing}. "
            f"Available columns begin: {header[:20]}"
        )

    connector_fields = [
        c for c in header if re.match(r"^(Steckertypen|Nennleistung Stecker)\s*\d+$", c)
    ]
    used = set(ESSENTIAL_REGISTRY_COLUMNS + OTHER_REGISTRY_COLUMNS + connector_fields)
    selected_columns = [c for c in header if c in used]

    matches = []
    rows_read = 0
    print(
        f"BNetzA header: line {header_index + 1}; encoding: {encoding}; "
        f"matching {len(candidates)} station IDs"
    )
    for chunk in pd.read_csv(
        path,
        sep=";",
        skiprows=header_index,
        header=0,
        encoding=encoding,
        dtype="string",
        usecols=selected_columns,
        chunksize=chunk_size,
        keep_default_na=True,
        low_memory=False,
    ):
        chunk.columns = chunk.columns.str.strip()
        rows_read += len(chunk)
        ids = chunk["Ladeeinrichtungs-ID"].map(normalize_id)
        subset = chunk.loc[ids.isin(candidates)].copy()
        if not subset.empty:
            subset.insert(0, "station_id", ids.loc[subset.index].to_numpy())
            matches.append(subset)

    if not matches:
        # Return an empty table with a stable station_id column.
        return pd.DataFrame(columns=["station_id"] + selected_columns), rows_read, connector_fields
    return pd.concat(matches, ignore_index=True), rows_read, connector_fields


def equipment_class(type_text: object, connector_text: str) -> str:
    s = ("" if pd.isna(type_text) else str(type_text)).lower()
    c = connector_text.lower()
    if "schnell" in s or any(x in c for x in ("ccs", "chademo", "dc ")):
        return "FAST_CHARGING_LIKELY"
    if "normal" in s or "ac typ" in c:
        return "NORMAL_CHARGING_LIKELY"
    return "UNKNOWN_HARDWARE_TYPE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Join real occupancy stations to BNetzA equipment records")
    parser.add_argument("--chunksize", type=int, default=30000)
    args = parser.parse_args()
    if args.chunksize < 100:
        parser.error("--chunksize must be at least 100")

    candidates, candidate_filename = read_candidates()
    registry_files = sorted(EXTERNAL.glob("Ladesaeulenregister_BNetzA*.csv"))
    if not registry_files:
        raise FileNotFoundError(
            "No Ladesaeulenregister_BNetzA*.csv found in data/external."
        )
    register_file = registry_files[-1]
    records, rows_read, connector_fields = read_registry_matches(
        register_file, set(candidates["station_id"]), args.chunksize
    )

    if records.empty:
        match_counts = pd.DataFrame(columns=["station_id", "registry_rows_for_id"])
        first = records.drop(columns=["Ladeeinrichtungs-ID"], errors="ignore")
    else:
        match_counts = (
            records.groupby("station_id", as_index=False)
            .size().rename(columns={"size": "registry_rows_for_id"})
        )
        first = records.drop_duplicates("station_id").drop(
            columns=["Ladeeinrichtungs-ID"], errors="ignore"
        )

    result = candidates.merge(first, on="station_id", how="left", validate="one_to_one")
    result = result.merge(match_counts, on="station_id", how="left", validate="one_to_one")
    result["registry_rows_for_id"] = result["registry_rows_for_id"].fillna(0).astype(int)

    count_raw = result["Anzahl Ladepunkte"] if "Anzahl Ladepunkte" in result else pd.Series(pd.NA, index=result.index)
    power_raw = (
        result["Nennleistung Ladeeinrichtung [kW]"]
        if "Nennleistung Ladeeinrichtung [kW]" in result
        else pd.Series(pd.NA, index=result.index)
    )
    result["registered_charging_points"] = number(count_raw)
    result["registered_equipment_power_kw"] = number(power_raw)

    # Keep charging-point count agreement separate from ID agreement.
    observed_counts = (
        number(result["typical_total_points"])
        if "typical_total_points" in result
        else pd.Series(float("nan"), index=result.index)
    )
    result["observed_typical_points"] = observed_counts

    states = []
    for _, r in result.iterrows():
        n = r["registry_rows_for_id"]
        official = r["registered_charging_points"]
        observed = r["observed_typical_points"]
        if n == 0:
            states.append("NO_REGISTRY_ID_MATCH")
        elif n > 1:
            states.append("DUPLICATE_REGISTRY_ID_REVIEW")
        elif pd.isna(official) or pd.isna(observed):
            states.append("ID_MATCH_POINTS_NOT_COMPARABLE")
        elif abs(float(official) - float(observed)) > 0.01:
            states.append("ID_MATCH_POINT_COUNT_DIFFERS")
        else:
            states.append("ID_AND_POINT_COUNT_MATCH")
    result["infrastructure_match_status"] = states

    connector_type_fields = [c for c in connector_fields if c.startswith("Steckertypen")]
    classes = []
    for _, r in result.iterrows():
        text = " ".join(
            str(r[c]) for c in connector_type_fields
            if c in result.columns and pd.notna(r[c])
        )
        classes.append(equipment_class(r.get("Art der Ladeeinrichtung", ""), text))
    result["equipment_class_for_screening"] = classes
    result["dataset_time_alignment"] = (
        "BNetzA register snapshot and occupancy observation dates may differ"
    )

    summary = (
        result.groupby(["infrastructure_match_status", "equipment_class_for_screening"], dropna=False)
        .size().reset_index(name="station_count")
        .sort_values("station_count", ascending=False)
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "german_occupancy_infrastructure_matches.csv"
    summary_path = PROCESSED / "german_occupancy_infrastructure_match_summary.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n--- REAL GERMAN OCCUPANCY / EQUIPMENT MATCH ---")
    print(f"Candidate source: {candidate_filename} | Stations: {len(candidates)}")
    print(f"BNetzA source: {register_file.name} | Registry rows inspected: {rows_read:,}")
    print(f"Station IDs found: {(result['registry_rows_for_id'] > 0).sum()} / {len(result)}")
    print("\nMatch status summary:")
    print(result["infrastructure_match_status"].value_counts().to_string())
    print("\nEquipment classes in matched sites:")
    print(
        result.loc[result["registry_rows_for_id"] > 0, "equipment_class_for_screening"]
        .value_counts().to_string()
    )
    print("\nFirst 10 candidate sites:")
    shown = [
        "station_id", "observed_typical_points", "registered_charging_points",
        "registered_equipment_power_kw", "equipment_class_for_screening",
        "infrastructure_match_status",
    ]
    print(result[shown].head(10).to_string(index=False))
    print(f"\nSaved: {output}")
    print(f"Saved: {summary_path}")
    print(
        "NOTE: Equipment IDs/counts are for infrastructure comparison only. "
        "No actual grid capacity, rejected arrivals or charging-session energy "
        "is inferred."
    )


if __name__ == "__main__":
    main()
