from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Source" / "weatherAUS.csv"
DEFAULT_OUTPUT = ROOT / "Output" / "weatherAUS_hci_scores.csv"
DEFAULT_WORKBOOK = ROOT / "Source" / "weatherAUS.xlsx"

REQUIRED_COLUMNS = {
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
}


def score_temperature(data: pd.DataFrame) -> pd.Series:
    maximum = pd.to_numeric(data["MaxTemp"], errors="coerce")
    minimum = pd.to_numeric(data["MinTemp"], errors="coerce")
    # Peak comfort is a 23-26 C maximum; hotter and colder bands score lower.
    # np.select uses the first match, so list the maximum-temperature thresholds high to low.
    conditions = [
        maximum.ge(39),
        maximum.ge(37),
        maximum.ge(35),
        maximum.ge(33),
        maximum.ge(31),
        maximum.ge(29),
        maximum.ge(27),
        maximum.ge(26),
        maximum.ge(23),
        maximum.ge(20),
        maximum.ge(18),
        maximum.ge(15),
        maximum.ge(11),
        maximum.ge(7),
        maximum.ge(0),
    ]
    choices = [0, 2, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4]
    scores = pd.Series(np.select(conditions, choices, default=2), index=data.index)
    # A minimum below 8 C removes one point; missing MinTemp means no penalty.
    scores = scores.sub(minimum.lt(8).fillna(False).astype(int)).clip(lower=0)
    # Without MaxTemp there is no temperature band to score.
    return scores.where(maximum.notna()).astype(float)


def score_precipitation(data: pd.DataFrame) -> pd.Series:
    rainfall = pd.to_numeric(data["Rainfall"], errors="coerce")
    # Dry days score best; points fall by rainfall band, with 25 mm or more penalized.
    conditions = [
        rainfall.eq(0),
        rainfall.lt(3),
        rainfall.lt(6),
        rainfall.lt(12),
        rainfall.lt(25),
    ]
    scores = pd.Series(
        np.select(conditions, [10, 8, 5, 2, 0], default=-2), index=data.index
    )
    return scores.where(rainfall.notna()).astype(float)


def score_sunshine(data: pd.DataFrame) -> pd.Series:
    sunshine = pd.to_numeric(data["Sunshine"], errors="coerce")
    evaporation = pd.to_numeric(data["Evaporation"], errors="coerce")
    # Sunshine is the primary measure; evaporation is used only when sunshine is missing.
    sunshine_conditions = [
        sunshine.ge(9),
        sunshine.ge(7.5),
        sunshine.ge(6),
        sunshine.ge(4.5),
        sunshine.ge(3),
        sunshine.ge(1.5),
    ]
    sunshine_score = pd.Series(
        np.select(sunshine_conditions, [10, 9, 8, 6, 4, 2], default=1),
        index=data.index,
    ).where(sunshine.notna())

    evaporation_conditions = [
        evaporation.ge(7),
        evaporation.ge(5.5),
        evaporation.ge(4),
        evaporation.ge(2.5),
    ]
    evaporation_score = pd.Series(
        np.select(evaporation_conditions, [10, 8, 6, 4], default=2),
        index=data.index,
    ).where(evaporation.notna())
    # If both measures are missing, leave the component missing for peer-median fallback.
    return sunshine_score.fillna(evaporation_score).astype(float)


def score_wind(data: pd.DataFrame) -> pd.Series:
    gust = pd.to_numeric(data["WindGustSpeed"], errors="coerce")
    # Lower gusts score better; gusts of 80 km/h or more receive a negative score.
    conditions = [
        gust.lt(30),
        gust.lt(40),
        gust.lt(50),
        gust.lt(63),
        gust.lt(80),
    ]
    scores = pd.Series(
        np.select(conditions, [10, 8, 6, 3, 0], default=-5), index=data.index
    )
    return scores.where(gust.notna()).astype(float)


def load_location_type_lookup(workbook_path: Path) -> dict[str, str]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Reference workbook not found: {workbook_path}")

    mapping = pd.read_excel(
        workbook_path,
        sheet_name="Location Population",
        usecols=["Location", "Location Type"],
    )
    if mapping["Location Type"].isna().any():
        raise ValueError(
            "The workbook has blank Location Type results. "
            "Open and save it in Excel to refresh its formula values."
        )

    # Normalize lookup keys so casing or surrounding spaces do not prevent a match.
    location_keys = mapping["Location"].astype("string").str.strip().str.casefold()
    location_types = mapping["Location Type"].astype("string").str.strip()
    return dict(zip(location_keys, location_types))


def fill_with_other_location_medians(
    scores: pd.Series,
    dates: pd.Series,
    locations: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    scored_rows = pd.DataFrame(
        {"date": dates, "location": locations, "score": scores}
    )
    # Use observed scores only, so imputed values never feed either fallback pool.
    daily_location_scores = (
        scored_rows.dropna(subset=["score"])
        .groupby(["date", "location"], sort=False)["score"]
        .median()
    )
    # The broader pool is the median of each other location's observed daily medians.
    scores_by_date = {
        date: group.droplevel("date")
        for date, group in daily_location_scores.groupby(level="date", sort=False)
    }
    location_medians = daily_location_scores.groupby(level="location").median()

    # Resolve each missing date/location once, then apply its fallback to every matching row.
    missing_keys = (
        scored_rows.loc[scores.isna(), ["date", "location"]]
        .drop_duplicates()
    )
    fallback_rows = []
    for date, missing_for_date in missing_keys.groupby("date", sort=False):
        same_date_scores = scores_by_date.get(date)
        for location in missing_for_date["location"].unique():
            value = np.nan
            method = "unavailable"

            # Prefer observed peers from that date; use the broader pool only if none exist.
            if same_date_scores is not None:
                peers = same_date_scores.drop(index=location, errors="ignore")
                if not peers.empty:
                    value = peers.median()
                    method = "same-date other-location median"

            if pd.isna(value):
                other_location_medians = location_medians.drop(
                    index=location, errors="ignore"
                )
                if not other_location_medians.empty:
                    value = other_location_medians.median()
                    method = "other-location median"

            fallback_rows.append((date, location, value, method))

    fallback_lookup = pd.DataFrame(
        fallback_rows,
        columns=["date", "location", "fallback_score", "fallback_method"],
    ).set_index(["date", "location"])
    row_keys = pd.MultiIndex.from_arrays(
        [dates, locations], names=["date", "location"]
    )
    # Reindexing restores the original row order and handles repeated date/location keys.
    aligned_fallbacks = fallback_lookup.reindex(row_keys)
    filled_scores = scores.where(
        scores.notna(), aligned_fallbacks["fallback_score"].to_numpy()
    )
    fallback_methods = pd.Series(
        aligned_fallbacks["fallback_method"].to_numpy(),
        index=scores.index,
        dtype="string",
    ).where(scores.isna())
    return filled_scores.astype(float), fallback_methods


def calculate_hci_scores(
    data: pd.DataFrame,
    location_type_lookup: dict[str, str],
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = data.copy()
    # Treat timestamps on the same calendar day as the same peer-comparison date.
    dates = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"Date contains {int(dates.isna().sum())} invalid or blank rows")

    locations = result["Location"].astype("string").str.strip()
    if locations.isna().any() or locations.eq("").any():
        raise ValueError("Location contains blank rows")
    # Use cleaned, case-insensitive keys for workbook lookup and peer matching.
    location_keys = locations.str.casefold()
    # Unmatched names are flagged for review rather than silently assigned a type.
    location_types = location_keys.map(location_type_lookup).fillna("Check location")
    if "Location Type" in result.columns:
        result["Location Type"] = location_types
    else:
        result.insert(result.columns.get_loc("Location") + 1, "Location Type", location_types)

    # Only sun and wind use peer medians; temperature and rainfall must be observed.
    result["T_Score"] = score_temperature(result)
    result["P_Score"] = score_precipitation(result)
    result["S_Score"], result["S_Score_Fallback"] = (
        fill_with_other_location_medians(score_sunshine(result), dates, location_keys)
    )
    result["W_Score"], result["W_Score_Fallback"] = (
        fill_with_other_location_medians(score_wind(result), dates, location_keys)
    )

    required_scores = result[["T_Score", "P_Score", "S_Score", "W_Score"]]
    has_all_scores = required_scores.notna().all(axis=1)
    # Weight components 4:3:2:1; any unavailable component leaves the composite blank.
    hci_score = (
        4 * result["T_Score"]
        + 3 * result["P_Score"]
        + 2 * result["S_Score"]
        + result["W_Score"]
    )
    result["HCI Score"] = hci_score.where(has_all_scores).clip(0, 100).round(1)

    # A scored day is favourable at 70 or above; missing composites remain unknown.
    favourable_day = pd.Series(pd.NA, index=result.index, dtype="Int64")
    has_hci_score = result["HCI Score"].notna()
    favourable_day.loc[has_hci_score] = (
        result.loc[has_hci_score, "HCI Score"].ge(70).astype("int64")
    )
    result["Favourable Day"] = favourable_day
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate workbook-matching HCI scores for weatherAUS.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    data = pd.read_csv(args.input, na_values=["NA", "N/A", ""])
    location_type_lookup = load_location_type_lookup(args.workbook)
    scored = calculate_hci_scores(data, location_type_lookup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    print(f"Saved scored dataset to {args.output}")
    print(f"Rows: {len(scored):,}")
    print(f"Location types: {scored['Location Type'].nunique()} categories")
    print(
        "Unmatched locations: "
        f"{scored['Location Type'].eq('Check location').sum():,} rows"
    )
    print(f"Rows with an HCI score: {scored['HCI Score'].notna().sum():,}")
    print(f"Rows without an HCI score: {scored['HCI Score'].isna().sum():,}")
    for score_name in ("S_Score", "W_Score"):
        fallback_count = scored[f"{score_name}_Fallback"].notna().sum()
        unresolved_count = scored[score_name].isna().sum()
        print(
            f"{score_name}: {fallback_count:,} fallback values; "
            f"{unresolved_count:,} still unavailable"
        )


if __name__ == "__main__":
    main()