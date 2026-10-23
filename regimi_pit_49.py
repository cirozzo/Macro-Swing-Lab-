#!/usr/bin/env python3
"""
MACRO SWING LAB 4.9
Ricostruzione dei regimi macroeconomici point-in-time
tramite FRED / ALFRED.

Serie:
- INDPRO: produzione industriale
- PCEPILFE: Core PCE

Output:
output/regimi_pit_49.csv

Non modifica i risultati delle versioni precedenti.
"""

import json
import os
import random
import time

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

START = "2007-01-01"

SERIES = (
    "INDPRO",
    "PCEPILFE"
)

BATCH_SIZE = 20

OUT = Path("output")

API_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


# ============================================================
# DOWNLOAD DATI ALFRED
# ============================================================

def fetch_batch(series, dates, key):

    params = {
        "series_id": series,
        "api_key": key,
        "file_type": "json",
        "output_type": 2,
        "vintage_dates": ",".join(dates),
        "observation_start": "2005-01-01",
        "observation_end": dates[-1],
        "limit": 100000,
    }

    url = API_URL + "?" + urlencode(params)

    for attempt in range(9):

        try:

            request = Request(
                url,
                headers={
                    "User-Agent": "MacroSwingLab-PIT/4.9"
                }
            )

            with urlopen(request, timeout=90) as response:
                payload = json.load(response)

            if "observations" not in payload:
                raise RuntimeError(
                    f"Invalid ALFRED payload for {series}"
                )

            if int(payload.get("count", 0)) > len(
                payload["observations"]
            ):
                raise RuntimeError(
                    f"Truncated ALFRED response for {series}"
                )

            return payload["observations"]

        except HTTPError as exc:

            if exc.code not in (
                429, 500, 502, 503, 504
            ) or attempt == 8:

                raise RuntimeError(
                    f"ALFRED HTTP {exc.code} "
                    f"for {series}, batch "
                    f"{dates[0]}..{dates[-1]}"
                ) from None

            retry_after = exc.headers.get("Retry-After")

            try:
                delay = (
                    min(180, float(retry_after))
                    if retry_after
                    else min(180, 8 * 2**attempt)
                )
            except ValueError:
                delay = min(180, 8 * 2**attempt)

        except (TimeoutError, URLError) as exc:

            if attempt == 8:
                raise RuntimeError(
                    f"ALFRED connection failure "
                    f"for {series}: {exc}"
                ) from None

            delay = min(180, 8 * 2**attempt)

        print(
            f"ALFRED retry {attempt + 1}/9 "
            f"for {series}; sleeping {delay:.0f}s",
            flush=True
        )

        time.sleep(
            delay + random.uniform(0, 2)
        )

    raise RuntimeError(
        "Unexpected retry exhaustion"
    )


# ============================================================
# DECODIFICA VINTAGE STORICHE
# ============================================================

def decode_batch(observations, dates, series):

    table = pd.DataFrame(observations)

    if table.empty:
        return {
            vintage: pd.Series(dtype="float64")
            for vintage in dates
        }

    if "date" not in table.columns:
        raise RuntimeError(
            f"Missing observation date "
            f"in {series} batch"
        )

    table["date"] = pd.to_datetime(
        table["date"]
    )

    result = {}

    for vintage in dates:

        # CORREZIONE:
        # ALFRED restituisce INDPRO_20070104
        # anziché 2007-01-04.

        candidates = (
            f"{series}_{vintage.replace('-', '')}",
            vintage,
            vintage.replace("-", "")
        )

        column_name = next(
            (
                col for col in candidates
                if col in table.columns
            ),
            None
        )

        if column_name is None:

            raise RuntimeError(
                f"ALFRED did not return vintage "
                f"{vintage} for {series}; "
                f"expected {candidates}; "
                f"columns={list(table.columns)[:8]}"
            )

        values = pd.to_numeric(
            table[column_name],
            errors="coerce"
        )

        result[vintage] = pd.Series(
            values.to_numpy(),
            index=table["date"]
        ).sort_index().dropna()

    return result


# ============================================================
# CALCOLO INDICATORI MACROECONOMICI
# ============================================================

def feature(values):

    if values.empty:
        return None, None

    # Variazione percentuale su 12 mesi.

    yoy = (
        values.pct_change(
            12,
            fill_method=None
        ) * 100
    )

    # Variazione del tasso YoY su 3 mesi.

    change = yoy.diff(3).dropna()

    if change.empty:
        return None, None

    return (
        float(change.iloc[-1]),
        change.index[-1].strftime("%Y-%m-%d")
    )


# ============================================================
# CLASSIFICAZIONE REGIMI R1-R4
# ============================================================

def classify(growth, inflation):

    if growth is None or inflation is None:
        return "SCONOSCIUTO"

    if growth >= 0 and inflation < 0:
        return "R1_CRESCITA_DISINFLAZIONE"

    if growth >= 0 and inflation >= 0:
        return "R2_CRESCITA_INFLAZIONE"

    if growth < 0 and inflation >= 0:
        return "R3_RALLENTAMENTO_INFLAZIONE"

    return "R4_RALLENTAMENTO_DISINFLAZIONE"


# ============================================================
# PROGRAMMA PRINCIPALE
# ============================================================

def main():

    key = os.getenv(
        "FRED_API_KEY", ""
    ).strip()

    if not key:
        raise RuntimeError(
            "Missing FRED_API_KEY "
            "GitHub Actions secret"
        )

    OUT.mkdir(
        exist_ok=True
    )

    snapshots = pd.date_range(
        START,
        pd.Timestamp.now(tz="UTC").date(),
        freq="W-FRI"
    )

    pairs = [
        (
            day.strftime("%Y-%m-%d"),
            (
                day - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        )
        for day in snapshots
    ]

    vintages = [
        vintage
        for _, vintage in pairs
    ]

    snapshots_by_series = {
        series: {}
        for series in SERIES
    }

    calls = 0

    # Download in blocchi per ridurre
    # il numero di richieste API.

    for start in range(
        0,
        len(vintages),
        BATCH_SIZE
    ):

        chunk = vintages[
            start:start + BATCH_SIZE
        ]

        for series in SERIES:

            raw = fetch_batch(
                series,
                chunk,
                key
            )

            decoded = decode_batch(
                raw,
                chunk,
                series
            )

            snapshots_by_series[
                series
            ].update(decoded)

            calls += 1

            time.sleep(1.2)

        print(
            f"Fetched "
            f"{min(start + BATCH_SIZE, len(vintages))}"
            f"/{len(vintages)} weekly vintages "
            f"({calls} requests)",
            flush=True
        )

    # ========================================================
    # COSTRUZIONE DATASET SETTIMANALE
    # ========================================================

    rows = []

    for decision, vintage in pairs:

        growth, growth_date = feature(
            snapshots_by_series[
                "INDPRO"
            ][vintage]
        )

        inflation, inflation_date = feature(
            snapshots_by_series[
                "PCEPILFE"
            ][vintage]
        )

        regime = classify(
            growth,
            inflation
        )

        rows.append({

            "data_segnale": decision,

            "vintage_asof": vintage,

            "growth_change": growth,

            "inflation_change": inflation,

            "growth_last_observation": growth_date,

            "inflation_last_observation": inflation_date,

            "regime_pit_49": regime

        })

    result = pd.DataFrame(rows)

    # ========================================================
    # CONTROLLI DI QUALITÀ
    # ========================================================

    assert result.data_segnale.is_unique

    assert (
        result.vintage_asof
        < result.data_segnale
    ).all()

    # ========================================================
    # SALVATAGGIO
    # ========================================================

    output_file = (
        OUT / "regimi_pit_49.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved {len(result)} weekly PIT "
        f"snapshots with {calls} API requests"
    )

    print(
        result.regime_pit_49
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "Diagnostic only: "
        "not yet joined to ETF outcomes."
    )


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    main()
