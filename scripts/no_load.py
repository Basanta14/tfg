"""Compare a no-load sweep with the healthy V/Hz reference."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tfg.helpers import operating_points

ROOT = Path(__file__).resolve().parent.parent
HEALTHY_SWEEP = ROOT / "data" / "sano" / "vacio sano.csv"
REFERENCE = ROOT / "data" / "puntos" / "referencia_vacio.csv"
REFERENCE_HEADER = (
    "# Healthy no-load. Sweep data/sano/vacio sano.csv. "
    "Do not replace it with another test.\n"
)

ALARM_PCT = -2.0
ALMOST_ALL = 0.8
MIN_HERTZ = 4
NO_LOAD_CURRENT = 1.0
FLAT_PCT = 0.05

_CURVE_COLUMNS = ["hertz", "frequency_hz", "voltage_v", "volts_per_hertz", "readings"]


def nearest_hertz(frequency: pd.Series) -> np.ndarray:
    """Round half up."""
    return np.floor(np.asarray(frequency, dtype=float) + 0.5).astype(int)


def by_hertz(points: pd.DataFrame) -> pd.DataFrame:
    """Median V/Hz per hertz."""
    if points.empty:
        return pd.DataFrame(columns=_CURVE_COLUMNS)

    frame = points.assign(
        hertz=nearest_hertz(points["f"]),
        volts_per_hertz=points["U"] / points["f"],
    )
    curve = frame.groupby("hertz", as_index=False, sort=True).agg(
        frequency_hz=("f", "median"),
        voltage_v=("U", "median"),
        volts_per_hertz=("volts_per_hertz", "median"),
        readings=("n", "sum"),
    )
    curve["hertz"] = curve["hertz"].astype(int)
    curve["readings"] = curve["readings"].astype(int)
    return curve


def no_load_stretches(points: pd.DataFrame) -> pd.DataFrame:
    return points.loc[points["I"] <= NO_LOAD_CURRENT]


def save_reference() -> pd.DataFrame:
    """Always the healthy sweep, never the log under test."""
    curve = by_hertz(no_load_stretches(operating_points(HEALTHY_SWEEP)))
    if curve.empty:
        raise SystemExit(f"The healthy sweep has no no-load stretches: {HEALTHY_SWEEP}")
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(REFERENCE_HEADER + curve.to_csv(index=False), encoding="utf-8")
    return curve


def read_reference() -> pd.DataFrame:
    if not REFERENCE.is_file():
        print("There was no reference. Saving it from the healthy no-load test.\n")
        return save_reference()
    return pd.read_csv(REFERENCE, comment="#")


def compare(curve: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    healthy = reference[["hertz", "volts_per_hertz"]].rename(
        columns={"volts_per_hertz": "volts_per_hertz_healthy"}
    )
    joined = curve.merge(healthy, on="hertz", how="inner")
    if joined.empty:
        return joined
    now = joined["volts_per_hertz"]
    before = joined["volts_per_hertz_healthy"]
    joined["difference_pct"] = 100.0 * (now - before) / before
    return joined.sort_values("hertz").reset_index(drop=True)


def verdict(table: pd.DataFrame) -> tuple[str, str, int]:
    """Title, sentence, and status: 2 demagnetized, 1 inconclusive, 0 healthy."""
    n = len(table)
    median = float(table["difference_pct"].median())
    below = int((table["difference_pct"] < 0).sum())
    whole_curve = below >= ALMOST_ALL * n

    if n < MIN_HERTZ:
        return (
            "Cannot tell.",
            f"Only {n} hertz match the reference. "
            "A no-load test that covers several frequencies is required.",
            1 if median <= ALARM_PCT else 0,
        )
    if median <= ALARM_PCT and whole_curve:
        return (
            "Looks demagnetized.",
            f"Median {median:+.1f} %. {below} of {n} hertz are below the reference. "
            "The whole curve has dropped.",
            2,
        )
    if median <= ALARM_PCT:
        return (
            "The test should be repeated.",
            f"The median is {median:+.1f} %, but the drop is not across the whole curve.",
            1,
        )
    if median < -1.0:
        return (
            "Still healthy.",
            f"Median {median:+.1f} %. The drop is small. It does not reach the alarm.",
            0,
        )
    if median > 1.0:
        return (
            "Still healthy.",
            f"Median {median:+.1f} %. It is above the reference. This is not demagnetization.",
            0,
        )
    return (
        "Still healthy.",
        f"Median {median:+.1f} %. The curve has not moved.",
        0,
    )


def show_table(table: pd.DataFrame) -> None:
    print(f"{'hertz':>6}  {'now V/Hz':>10}  {'healthy V/Hz':>12}  {'difference':>12}")
    for row in table.itertuples(index=False):
        percentage = 0.0 if abs(row.difference_pct) < FLAT_PCT else row.difference_pct
        difference = f"{percentage:+.1f} %"
        print(
            f"{row.hertz:6d}  "
            f"{row.volts_per_hertz:10.3f}  "
            f"{row.volts_per_hertz_healthy:12.3f}  "
            f"{difference:>12}"
        )


def analyze(log_path: Path, reference: pd.DataFrame, title: str) -> int:
    points = operating_points(log_path)
    print(title)
    if points.empty:
        print(
            "There is no steady stretch. "
            "The frequency must be stable, with at least 30 readings and a steady voltage."
        )
        return 1

    no_load = no_load_stretches(points)
    loaded = len(points) - len(no_load)
    if loaded:
        print(f"Leaving out {loaded} stretches with current: that is not no-load.")
    if no_load.empty:
        print("This log is not a no-load test.")
        return 1

    curve = by_hertz(no_load)
    unmatched = sorted(set(curve["hertz"]) - set(reference["hertz"]))
    table = compare(curve, reference)
    if table.empty:
        print("No hertz in this test is present in the reference.")
        return 1

    print(f"{len(table)} hertz match the reference.")
    show_table(table)
    if unmatched:
        outside = ", ".join(str(hertz) for hertz in unmatched)
        print(f"No match in the reference: {outside} Hz.")
    verdict_title, sentence, code = verdict(table)
    print()
    print(verdict_title)
    print(sentence)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a new no-load test with the healthy no-load reference."
    )
    parser.add_argument(
        "log",
        nargs="?",
        help="CSV of a no-load sweep, from 45 to 65 Hz.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Save the reference again from the healthy sweep.",
    )
    args = parser.parse_args()

    if args.reference:
        print(
            "This replaces the fingerprint. Only if the sweep being saved is healthy.\n"
        )
        reference = save_reference()
    else:
        reference = read_reference()

    print(
        f"Reference: {len(reference)} hertz, "
        f"from {int(reference['hertz'].min())} to {int(reference['hertz'].max())} Hz. "
        f"Saved in {REFERENCE.relative_to(ROOT)}."
    )
    print()

    if not args.log:
        print("Pass the CSV of a no-load sweep to compare it with this reference.")
        raise SystemExit(0)
    raise SystemExit(analyze(Path(args.log), reference, Path(args.log).name))


if __name__ == "__main__":
    main()
