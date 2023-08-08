from __future__ import annotations

import argparse
import calendar
import importlib
import os
import pathlib
import sys
from datetime import datetime
from typing import Sequence

import pandas as pd
from pandas import to_datetime as pd_to_dt

if sys.version_info <= (3, 9):
    from typing import Union
    DataFrameSheets = dict[Union[int, str], pd.DataFrame]
else:
    from typing import TypeAlias
    DataFrameSheets: TypeAlias = dict[int | str, pd.DataFrame]

VERSION = importlib.metadata.version('tcc_stats')
CLI_WIDTH = os.get_terminal_size().columns

CWD = pathlib.Path(os.getcwd())
YEARS = sorted(list(range(2018, datetime.now().year + 1)))
MONTHS = [calendar.month_name[i] for i in range(1, 13)]
ALLOWED_EXCEL = ('.xlsx', '.xls', '.xlsm', '.xlsb')
COLS = ('operator', 'requester_code', 'doc_type', 'fdr_no', 'result', 'pdf',
        'minutes', 'source_lang', 'target_lang')

DESCRIPTION = """\
Plot TCC statistics for:
    - the current year
    - previous years (by providing the directory)
    - given month(s) of the current or previous years

The user can provide:

(a) no directory, so 'data/' will be used. The directory structure should be:

|-- data (or any specified name)
    |-- <year1>
        |-- <month1>.csv
        |-- <month2>.csv
        |-- <month3>.csv
    |-- <year2>
        |-- <all_months>.xlsx (max. one sheet per month)

(b) a path to a directory
(c) a list of years
(d) a list of months, `year` is the current one by default
(e) a list of years and months

It is not possible to provide both a path and any of the other arguments
at the same time.
"""

RED = '\033[41m'
GREEN = '\033[42m'
YELLOW = '\033[43;30m'
TURQUOISE = '\033[46;30m'
SUBTLE = '\033[2m'
NORMAL = '\033[m'


def colored_message(msg: str, color: str, end: str = NORMAL) -> str:
    """Color the background and text of a message."""
    return f"{color}{msg}{end}"


def step(
    message: str,
    *,
    start: str = '',
    color: str = NORMAL,
    cols: int = 80,
) -> None:
    """Print a message with a colored background and text."""
    message = colored_message(message, color)
    dots = "." * (cols - len(start) - len(message) + len(color))
    print(f"{start}{dots}{message}", end=None)


def get_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'path', type=str, default=pathlib.Path(CWD, 'data'),
        help=(
            "Path to the directory containing the statistics files inside "
            "individual directories for each year. (default: `./data/`)"
        )
    )

    parser.add_argument(
        '-y', '--years', type=int, nargs='+', default=YEARS,
        help=(
            "Year(s) to plot the statistics for. Should match the name of the "
            "directory inside of which the statistics files are located."
        )
    )

    parser.add_argument(
        '-m', '--months', type=parse_months, nargs='+',
        default=",".join([str(i) for i in range(1, 13)]),
        help=(
            "Month(s) to plot the statistics for. If not provided, "
            "all 12 months will be used."
        )
    )

    parser.add_argument(
        '-o', '--output', type=str, default='export.csv',
        help=(
            "Output file name (or relative or absolute path). If it does not "
            "exist, the (sub)directories will be created. Allowed extentions: "
            f"{ALLOWED_EXCEL + ('.csv',)}. (default: `%(default)s`)"
        )
    )

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Print more information about the data along with a small sample."
    )

    # https://stackoverflow.com/a/8521644/812183
    parser.add_argument(
        '-V', '--version', action='version',
        version=f'{SUBTLE}%(prog)s {VERSION}{NORMAL}',
    )

    return parser.parse_args()


def parse_months(months: str | None) -> int | Sequence[int]:
    """Parse the `--months` argument.
    `--months` can be:
      - None -> all 12 months
      - 1 -> January, 2 -> February, ..., 12 -> December
      - any number of months between 1 and 12 separated by a comma
    """
    parsed_month = None
    parsed_months = []

    if months is None:
        parsed_months = list(range(1, 13))
    elif ',' in months:
        months_list = [int(month) for month in months.split(',')]
        if any(month == 0 for month in months_list) and len(months_list) > 1:
            print("Will raise an error.")
            raise argparse.ArgumentTypeError(
                "Months cannot be 0 and any other number at the same time."
            )
        parsed_months = [int(month) for month in months_list]
        parsed_months = sorted(list(set(parsed_months)))
    elif isinstance(months, str):
        parsed_month = int(months)
    else:
        raise argparse.ArgumentTypeError(
            f"{months} is not a valid type for the months argument."
        )
    return parsed_month or parsed_months


def preprocess_path(path: str | pathlib.Path | None) -> pathlib.Path:
    """Preprocess the path to the directory containing the statistics files."""
    if path is None:
        path = pathlib.Path(CWD, 'data')
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
        if path.suffix in ALLOWED_EXCEL:
            df = pd.read_excel(path, usecols='A:I', index_col=None, names=COLS,
                               sheet_name=None)  # get all sheets, filter later
            step("[Opening]", start=filename, color=YELLOW)
        elif path.suffix == '.csv':
            df_ = pd.read_csv(path, index_col=None, names=COLS,
                              encoding='utf-8', sep=';')
            step("Done", start=filename, color=GREEN)
    except FileNotFoundError:
        step("(none)", start=filename, color=TURQUOISE)

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

    for i, month_name in enumerate(MONTHS, start=1):
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

    years = sorted(years) if years is not None else YEARS
    if months is None or months == 0:
        months_list = {i+1: month_name for i, month_name in enumerate(MONTHS)}
    elif isinstance(months, Sequence):
        # check if months is a list of lists and flatten it
        if any(isinstance(month, list) for month in months):
            months = [month for subl in months
                      for month in (subl if isinstance(subl, list) else [subl])]
        # months_list should be a list of tuples (month, month_name)
        # with only the specified months
        months_list = {i+1: month_name for i, month_name in enumerate(MONTHS)
                       if i+1 in months}
    elif isinstance(months, int):
        months_list = {months: MONTHS[months-1]}
    else:
        raise TypeError(f"{months!r} is not a valid type.")

    df = pd.DataFrame()

    for year in years:
        filename = f"{year}_stats.xlsx"
        filepath = pathlib.Path(path, str(year), filename)
        df_sheets = get_data(filepath)
        if df_sheets is not None:
            for i, month_name in enumerate(MONTHS, start=1):
                months_dict: dict[str | int, int] = {v: k for k, v
                                                     in months_list.items()}
                if i not in months_dict.values():
                    continue
                else:
                    sheetname = f"ANTE - {month_name.upper()} {year}"
                    # If `df`` comes from Excel, it is a dict of DataFrames
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
                    step("Done", start=f"{filename}/{sheetname}", color=GREEN)
            step("[Closing]", start=filename, color=YELLOW)

    if verbose:
        df.info()
        print()
        sample_size = min(10, len(df))
        print(df.sample(sample_size))

    return df


def preprocess_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Perform preprocessing of the data for statistical analysis."""

    if verbose:
        print("=" * CLI_WIDTH, "Initial DataFrame:", sep="\n", end="\n\n")
        print("-" * CLI_WIDTH, "INFO:", "=" * CLI_WIDTH, sep="\n")
        df.info()
        print()
        print("-" * CLI_WIDTH, "SAMPLE:", "=" * CLI_WIDTH, sep="\n")
        sample_size = min(10, len(df))
        print(df.sample(sample_size))

    # Remove unnecessary columns
    df = df.drop(['operator', 'fdr_no', 'source_lang', 'target_lang'], axis=1)

    # Convert 'requester_code' and 'doc_type' to categorical
    df['requester_code'] = df['requester_code'].astype('category')
    df['doc_type'] = df['doc_type'].astype('category')

    # Convert 'result' and 'pdf' to boolean
    df['improved'] = df['result'].map(
        {'OK': True, 'Improved': False}).astype('bool')
    df = df.drop('result', axis=1)
    df['pdf'] = df['pdf'].map({'yes': True, 'no': False}).astype('bool')

    if verbose:
        print("=" * CLI_WIDTH, "Final DataFrame:", sep="\n", end="\n\n")
        print("-" * CLI_WIDTH, "INFO:", "=" * CLI_WIDTH, sep="\n")
        df.info()
        print()
        print("-" * CLI_WIDTH, "SAMPLE:", "=" * CLI_WIDTH, sep="\n")
        sample_size = min(10, len(df))
        print(df.sample(sample_size))

    return df


def main() -> int:
    args = get_args()
    path = preprocess_path(args.path)
    years = set(args.years)
    months = args.months
    verbose = args.verbose
    output = args.output

    df = pd.DataFrame()
    pd.set_option('display.width', CLI_WIDTH)

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

        if output.suffix in ALLOWED_EXCEL:
            df.to_excel(output, index=False, sheet_name='TCC')
        elif output.suffix == '.csv':
            df.to_csv(output, index=False, encoding='utf-8', sep=';')
        else:
            raise ValueError(
                f"""{output} does not have a valid file extension.
                Only Excel and CSV files are supported."""
            )
        print()
        step("Saved", start=f"Saving {output.absolute().relative_to(CWD)}",
             color=GREEN)
    return 0


if '__main__' == __name__:
    raise SystemExit(main())
