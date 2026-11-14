"""Macro Swing Lab: audit strutturale CFTC disaggregated FUTURES ONLY, WTI 067651.
Intervallo richiesto 2010-2026; 2009 escluso e non certificato.
NON certifica le date effettive di pubblicazione (PIT).
"""
import io
import json
import pathlib
import sys
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'audit_cot_wti'
OUT.mkdir(exist_ok=True)
CODE = '067651'
START, END = 2010, 2026
FIELDS = ['M_Money_Positions_Long_All', 'M_Money_Positions_Short_All', 'Open_Interest_All']


def download_year(year, session):
    urls = [
        f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip',
        f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_{year}.zip',
    ]
    errors = []
    for url in urls:
        for attempt in range(3):
            try:
                response = session.get(url, timeout=(20, 90))
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = [n for n in archive.namelist()
                             if n.lower().endswith(('.txt', '.csv')) and not n.startswith('__MACOSX/')]
                    if len(names) != 1:
                        raise ValueError(f'Numero file tabellari inatteso: {names}')
                    with archive.open(names[0]) as source:
                        frame = pd.read_csv(source, low_memory=False, dtype=str, encoding='utf-8-sig')
                frame.columns = frame.columns.astype(str).str.strip()
                if frame.columns.duplicated().any():
                    raise ValueError('Intestazioni duplicate nel file CFTC')
                return frame, url
            except (requests.RequestException, zipfile.BadZipFile, ValueError, UnicodeError, pd.errors.ParserError) as exc:
                errors.append(f'{url} tentativo {attempt + 1}: {type(exc).__name__}: {str(exc)[:180]}')
                if attempt < 2:
                    time.sleep(attempt + 1)
    raise RuntimeError(' | '.join(errors))


def parse_report_dates(frame):
    specs = [
        ('Report_Date_as_YYYY-MM-DD', ['%Y-%m-%d', '%Y/%m/%d']),
        ('Report_Date_as_YYYY_MM_DD', ['%Y_%m_%d', '%Y-%m-%d']),
        ('Report_Date_as_MM_DD_YYYY', ['%m/%d/%Y', '%m-%d-%Y', '%m_%d_%Y']),
        ('As_of_Date_In_Form_YYMMDD', ['%y%m%d']),
    ]
    attempted = []
    for column, formats in specs:
        if column not in frame.columns:
            continue
        raw = frame[column].astype('string').str.strip()
        for fmt in formats:
            parsed = pd.to_datetime(raw, format=fmt, errors='coerce')
            if parsed.notna().all():
                return parsed.dt.normalize(), column, fmt
        attempted.append(column)
    raise ValueError(f'Colonne data presenti ma non interamente interpretabili: {attempted}; '
                     f'colonne disponibili: {list(frame.columns)}')


def main():
    session = requests.Session()
    session.headers.update({'User-Agent': 'MacroSwingLab-COT-audit/3.0 (academic historical research)'})
    audit, frames = [], []
    for year in range(START, END + 1):
        row = {'year': year, 'status': 'ERROR', 'rows': 0, 'url': '', 'date_column': '',
               'date_format': '', 'market_names': '', 'error': ''}
        try:
            frame, url = download_year(year, session)
            row['url'] = url
            required = ['CFTC_Contract_Market_Code', *FIELDS]
            missing = [c for c in required if c not in frame.columns]
            if missing:
                raise ValueError(f'Campi obbligatori mancanti: {missing}')
            code = (frame['CFTC_Contract_Market_Code'].astype('string').str.strip()
                    .str.replace(r'\.0$', '', regex=True).str.zfill(6))
            frame = frame.loc[code.eq(CODE).fillna(False)].copy()
            if frame.empty:
                raise ValueError(f'Codice CFTC {CODE} assente')
            if 'FutOnly_or_Combined' in frame.columns:
                kinds = frame['FutOnly_or_Combined'].astype('string').str.strip().str.lower().unique().tolist()
                if not all('fut' in str(x) and 'comb' not in str(x) for x in kinds):
                    raise ValueError(f'Tipologia non futures-only: {kinds}')
            # La provenienza da fut_disagg_txt_ e il controllo schema sono obbligatori.
            dates, date_column, date_format = parse_report_dates(frame)
            frame['report_date'] = dates
            for field in FIELDS:
                frame[field] = pd.to_numeric(frame[field].str.replace(',', '', regex=False), errors='coerce')
            if frame[FIELDS].isna().any().any():
                raise ValueError('Posizioni o open interest non numerici')
            if (frame[FIELDS] < 0).any().any() or (frame[FIELDS[2]] <= 0).any():
                raise ValueError('Valori negativi o open interest nullo')
            if not frame.report_date.dt.year.eq(year).all():
                raise ValueError('Date report fuori anno archivio')
            if frame.report_date.duplicated().any():
                raise ValueError('Date duplicate per codice nello stesso anno: NON eliminate')
            frame['market'] = (frame['Market_and_Exchange_Names'].astype('string').str.strip()
                               if 'Market_and_Exchange_Names' in frame.columns else '')
            frame['source_year'], frame['source_url'] = year, url
            frames.append(frame[['report_date', 'market', *FIELDS, 'source_year', 'source_url']])
            row.update(status='OK', rows=len(frame), date_column=date_column,
                       date_format=date_format, market_names=' | '.join(frame.market.dropna().unique()))
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        audit.append(row)
        pd.DataFrame(audit).to_csv(OUT / 'audit_download.csv', index=False)
        print(f"{year}: {row['status']} {row['rows']} righe {row['error'][:200]}", flush=True)

    missing = [row['year'] for row in audit if row['status'] != 'OK']
    summary = {'contract_code': CODE, 'source': 'CFTC disaggregated futures-only yearly archives',
               'required_years': [START, END], 'missing_years': missing,
               'structural_status': 'INCOMPLETE',
               'pit_status': 'NOT CERTIFIED - real release dates not checked',
               'excluded_2009': '2009 intentionally outside required interval; NOT certified',
               'checked_at_utc': datetime.utcnow().isoformat() + 'Z'}
    if frames:
        data = pd.concat(frames, ignore_index=True).sort_values('report_date').reset_index(drop=True)
        data['net_mm'] = data[FIELDS[0]] - data[FIELDS[1]]
        data['net_pct_oi'] = data.net_mm / data[FIELDS[2]]
        data['delta_4w'] = data.net_pct_oi.diff(4)
        data['gap_days'] = data.report_date.diff().dt.days
        duplicates = data.report_date.duplicated(keep=False)
        irregular = data.gap_days.notna() & data.gap_days.ne(7)
        spikes = data.net_pct_oi.diff().abs().gt(.05)
        data.to_csv(OUT / 'cot_wti_067651_futures_only.csv', index=False)
        data.loc[duplicates].to_csv(OUT / 'date_duplicate.csv', index=False)
        data.loc[irregular].to_csv(OUT / 'settimane_irregolari.csv', index=False)
        data.loc[spikes].to_csv(OUT / 'salti_superiori_5pp.csv', index=False)
        summary.update(records=len(data), first=str(data.report_date.min().date()),
                       last=str(data.report_date.max().date()),
                       duplicate_rows=int(duplicates.sum()), irregular_gaps=int(irregular.sum()),
                       jumps_over_5pp=int(spikes.sum()),
                       market_names=sorted(data.market.dropna().unique().tolist()))
        if not missing and not duplicates.any() and not irregular.any() and data.net_pct_oi.notna().all():
            summary['structural_status'] = 'PASSED - not PIT certified'
        elif not missing:
            summary['structural_status'] = 'FAILED - review duplicate/irregular/invalid observations'
    (OUT / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['structural_status'] != 'PASSED - not PIT certified':
        sys.exit(1)


if __name__ == '__main__':
    main()
