from __future__ import annotations

import calendar
import importlib.metadata
import os
import pathlib
from datetime import datetime

VERSION = importlib.metadata.version('tcc_stats')

CLI_WIDTH = os.get_terminal_size().columns
SEP_DASH = '-' * CLI_WIDTH
SEP_EQ = '=' * CLI_WIDTH

ALLOWED_EXCEL = ('.xlsx', '.xls', '.xlsm', '.xlsb')
COMMANDS = ('extract', 'plot')

CWD = pathlib.Path(os.getcwd())

YEARS = sorted(list(range(2017, datetime.now().year + 1)))
MONTHS = [calendar.month_name[i] for i in range(1, 13)]
COLS = ('operator', 'requester_code', 'doc_type', 'fdr_no', 'result', 'pdf',
        'minutes', 'source_lang', 'target_lang')
TCCERS = {'JDP', 'JM', 'JVAU', 'MK', 'ND', 'SIL', 'VA'}

TITLE = ':bar_chart: :green[T]:red[C]:blue[C] Stats'

STATE = {
    'files': None,
}

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
