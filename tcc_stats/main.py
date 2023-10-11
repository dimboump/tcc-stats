from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import tcc_stats.constants as C
from tcc_stats.color import COLORS
from tcc_stats.commands.extract import extract
from tcc_stats.commands.plot import plot


def get_args() -> argparse.Namespace:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(
        description=C.DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter
    )

    # https://stackoverflow.com/a/8521644/812183
    parser.add_argument(
        '-V', '--version', action='version',
        version=f"{COLORS['subtle']}%(prog)s {C.VERSION}{COLORS['normal']}",
    )

    def add_cmd(name: str, *, help: str) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(name, help=help)
        return parser

    subparsers = parser.add_subparsers(dest='command')

    # 'extract' command
    extract_parser = add_cmd(
        'extract',
        help="Extract the statistics from Excel of csv files for: "
             "- the current year "
             "- any previous year(s) "
             "- given month(s) of the current or previous year(s)"
    )

    extract_parser.add_argument(
        'path', type=str, default=pathlib.Path(C.CWD, 'data'),
        help=(
            "Path to the directory containing the statistics files inside "
            "individual directories for each year. (default: `./data/`)"
        )
    )

    extract_parser.add_argument(
        '-y', '--years', type=int, nargs='+', default=C.YEARS,
        help=(
            "Year(s) to plot the statistics for. Should match the name of the "
            "directory inside of which the statistics files are located."
        )
    )

    extract_parser.add_argument(
        '-m', '--months', type=parse_months, nargs='+',
        default=",".join([str(i) for i in range(1, 13)]),
        help=(
            "Month(s) to plot the statistics for. If not provided, "
            "all 12 months will be used."
        )
    )

    extract_parser.add_argument(
        '-o', '--output', type=str, default='export.csv',
        help=(
            "Output file name (or relative or absolute path). If it does not "
            "exist, the (sub)directories will be created. Allowed extentions: "
            f"{C.ALLOWED_EXCEL + ('.csv',)}. (default: `%(default)s`)"
        )
    )

    extract_parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Print more information about the data along with a small sample."
    )

    # `plot` command
    plot_parser = add_cmd(
        'plot',
        help="Plot statistics using matplotlib."
    )

    plot_parser.add_argument(
        '-t', '--type', type=str, default='scatter',
        choices=['scatter', 'bar', 'line', 'pie'],
        help=(
            "Type of plot to use (choices: %(choices)s). "
            "(default: `%(default)s`)"
        )
    )

    plot_parser.add_argument(
        '-s', '--save', action='store_true',
        help="Save the plot to a file."
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


def main() -> int:

    args = get_args()

    if args.command == 'extract':
        return extract(args)

    elif args.command == 'plot':
        return plot(args)

    return 0


if '__main__' == __name__:
    raise SystemExit(main())
