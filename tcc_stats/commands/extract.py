from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime
from typing import Sequence

import pandas as pd
from pandas import to_datetime as pd_to_dt

from tcc_stats import color
from tcc_stats import constants as C

if sys.version_info <= (3, 9):
    from typing import Union
    DataFrameSheets = dict[Union[int, str], pd.DataFrame]
else:
    from typing import TypeAlias
    DataFrameSheets: TypeAlias = dict[int | str, pd.DataFrame]


def preprocess_path(path: str | pathlib.Path | None) -> pathlib.Path:
    """Preprocess the path to the directory containing the statistics files."""
    if path is None:
        path = pathlib.Path(C.CWD, 'data')
    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a valid directory.")
    return pathlib.Path(path)


def get_data(
    path: str | pathlib.Path,  # default: 'data/'
) -> pd.DataFrame | DataFrameSheets | None:
    """Read the data from the given file and return a DataFrame."""

    path = pathlib.Path(path)
    filename = path.name

    df, df_ = None, None

    try:
        if path.suffix in C.ALLOWED_EXCEL:
            df = pd.read_excel(path, usecols='A:I', index_col=None,
                               names=C.COLS, sheet_name=None)  # get all sheets
            color.step("[Opening]", start=filename, color='yellow')
        elif path.suffix == '.csv':
            df_ = pd.read_csv(path, index_col=None, names=C.COLS,
                              encoding='utf-8', sep=';')
            color.step("Done", start=filename, color='green')
    except FileNotFoundError:
        color.step("(none)", start=filename, color='turquoise')

    return df or df_


def get_current_data(
    path: str | pathlib.Path,  # default: 'data/'
    *,
    year: int = datetime.now().year,
    months: int | Sequence[int] = 0,
    verbose: bool = False
) -> pd.DataFrame:
    """Get a Pandas DataFrame of the data for the current year."""

    df = pd.DataFrame()

    for i, month_name in enumerate(C.MONTHS, start=1):
        if not isinstance(months, int) and i not in months:
            continue
        filename = f"{i}. {month_name.title()} stats all.csv"
        filepath = pathlib.Path(path, str(year), filename)
        df_temp = get_data(filepath)
        if df_temp is not None and isinstance(df_temp, pd.DataFrame):
            if isinstance(months, list):
                month = pd_to_dt(months[i-1], format='%m').month
            elif isinstance(months, int):
                month = pd_to_dt(i, format='%m').month
            df_temp = df_temp.assign(year=year, month=month)
            date = pd_to_dt(df_temp[['year', 'month']].assign(day=1),
                            format='%Y-%m')
            df_temp = df_temp.assign(date=date)
            df = pd.concat([df, df_temp], ignore_index=True)
        continue

    if verbose:
        df.info()
        print()
        sample_size = min(10, len(df))
        print(df.sample(sample_size))

    return df


def get_history_data(
    path: str | pathlib.Path,  # default: 'data/'
    *,
    years: Sequence[int] | set[int] | None = None,
    months: int | Sequence[int],
    verbose: bool = False
) -> pd.DataFrame:
    """Get a Pandas DataFrame of the data for the previous year(s)."""

    years = sorted(years) if years is not None else C.YEARS
    if months is None or months == 0:
        months_list = {i+1: month_name for i, month_name in enumerate(C.MONTHS)}
    elif isinstance(months, Sequence):
        # check if months is a list of lists and flatten it
        if any(isinstance(month, list) for month in months):
            months = [month for subl in months
                      for month in (subl if isinstance(subl, list) else [subl])]
        # months_list should be a list of tuples (month, month_name)
        # with only the specified months
        months_list = {i+1: month_name for i, month_name in enumerate(C.MONTHS)
                       if i+1 in months}
    elif isinstance(months, int):
        months_list = {months: C.MONTHS[months-1]}
    else:
        raise TypeError(f"{months!r} is not a valid type.")

    df = pd.DataFrame()

    for year in years:
        filename = f"{year}_stats.xlsx"
        filepath = pathlib.Path(path, str(year), filename)
        df_sheets = get_data(filepath)
        if df_sheets is not None:
            for i, month_name in enumerate(C.MONTHS, start=1):
                months_dict: dict[str | int, int] = {v: k for k, v
                                                     in months_list.items()}
                if i not in months_dict.values():
                    continue
                else:
                    sheetname = f"ANTE - {month_name.upper()} {year}"
                    # If `df` comes from Excel, it is a dict of DataFrames
                    # so we need to filter the sheets
                    if isinstance(df_sheets, dict):
                        df_temp = df_sheets[sheetname]
                    months_dict = {v: v for _, v in months_dict.items()}
                    month = pd_to_dt(months_dict[i], format='%m').month
                    df_temp = df_temp.assign(year=year, month=month)
                    date = pd_to_dt(df_temp[['year', 'month']].assign(day=1),
                                    format='%Y-%m')
                    df_temp = df_temp.assign(date=date)
                    df = pd.concat([df, df_temp], ignore_index=True)
                    color.step("Done", start=f"{filename}/{sheetname}",
                               color='green')
            color.step("[Closing]", start=filename, color='yellow')

    if verbose:
        df.info()
        print()
        sample_size = min(10, len(df))
        print(df.sample(sample_size))

    return df


def preprocess_data(
    df: pd.DataFrame,
    *,
    mark_trainees: bool = False,
    staff: set[str] | Sequence[str] = C.TCCERS,
    verbose: bool = False
) -> pd.DataFrame:
    """Perform preprocessing of the data for statistical analysis."""

    if verbose:
        print(C.SEP_EQ, "Initial DataFrame:", sep="\n", end="\n\n")
        print(C.SEP_DASH, "INFO:", C.SEP_EQ, sep="\n")
        df.info()
        print()
        print(C.SEP_DASH, "SAMPLE:", C.SEP_EQ, sep="\n")
        sample_size = min(5, len(df))
        print(df.sample(sample_size))

    # Remove unnecessary columns
    df = df.drop(['fdr_no', 'source_lang', 'target_lang'], axis=1)

    # Extract DG from 'requester_code'
    df['dg_code'] = df['requester_code'].str.split('-', n=1).str[0]

    # Convert columns to categorical
    df['operator'] = df['operator'].astype('category')
    df['requester_code'] = df['requester_code'].astype('category')
    df['dg_code'] = df['dg_code'].astype('category')
    df['doc_type'] = df['doc_type'].astype('category')

    if mark_trainees:
        df['trainee'] = ~df['operator'].isin(staff)

    # Convert 'result' and 'pdf' to boolean
    df['improved'] = df['result'].map(
        {'OK': False, 'Improved': True}).astype('bool')
    df = df.drop('result', axis=1)
    df['pdf'] = df['pdf'].map({'yes': True, 'no': False}).astype('bool')

    # Rearrange columns by splitting them right after requester_code
    # then dg_code, and then the rest, programmatically
    cols = df.columns.tolist()
    idx_before = cols.index('requester_code')
    cols = cols[:idx_before+1] + \
        ['dg_code'] + \
        [col for col in cols[idx_before+1:]
         if col not in ('requester_code', 'dg_code')]
    df = df[cols]

    if verbose:
        print(C.SEP_EQ, "Final DataFrame:", sep="\n", end="\n\n")
        print(C.SEP_DASH, "INFO:", C.SEP_EQ, sep="\n")
        df.info()
        print()
        print(C.SEP_DASH, "SAMPLE:", C.SEP_EQ, sep="\n")
        sample_size = min(5, len(df))
        print(df.sample(sample_size))

    return df


def extract(args: argparse.Namespace) -> int:
    path = preprocess_path(args.path)
    years = set(args.years)
    months = args.months
    verbose = args.verbose
    output = args.output

    df = pd.DataFrame()
    pd.set_option('display.width', C.CLI_WIDTH)

    current_year = datetime.now().year
    if current_year in years:
        df = get_current_data(path, months=months, verbose=verbose)
        years.remove(current_year)

    df = pd.concat([df, get_history_data(path, years=years, months=months,
                                         verbose=verbose)])
    df = preprocess_data(df, verbose=verbose)

    if output:
        output = pathlib.Path(output)
        if not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)

        if output.suffix in C.ALLOWED_EXCEL:
            df.to_excel(output, index=False, sheet_name='TCC')
        elif output.suffix == '.csv':
            df.to_csv(output, index=False, encoding='utf-8', sep=';')
        else:
            print(
                f"""{output} does not have a valid file extension.
                Only Excel and CSV files are supported."""
            )
            return 1
        print()
        relative_path = output.absolute().relative_to(C.CWD)
        color.step("Saved", start=f"Saving {relative_path}", color='green')
    return 0
