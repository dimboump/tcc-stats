# Statistics for TCC

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Create Release](https://github.com/dimboump/tcc-stats/actions/workflows/releases.yaml/badge.svg)](https://github.com/dimboump/tcc-stats/actions/workflows/releases.yaml)

## Installation and Usage

1. Clone the repository:

    ```bash
    git clone git@github.com:dimboump/tcc-stats
    ```

2. Install the dependencies and run the script:

   - Windows:

       ```bash
       pip install -r requirements.txt
       python tcc_stats.py
       ```

   - Linux/macOS:

       ```bash
       pip3 install -r requirements.txt
       python3 tcc_stats.py
       ```

Options:

| Option | Data Type | Description | Default |
| ------ | --------- | ----------- | ------- |
| `-p`, `--path` | `str` or `pathlib.Path` | Path to or name of the TCC data directory. | `data/` |
| `-y`, `--years` | `int`, `Sequence[int]` | Years to include in the statistics. You can specify a single year or multiple years, by separating them with a space. | 2018-today |
| `-m`, `--months` | `int`, `Sequence[int]` | Months to include in the statistics. To specify multiple months, separate them with a comma. | 12 months if the year is in the past, current month since January of this year otherwise |
| `-o`, `--output` | `str` or `pathlib.Path` | Path to or name of the output file (with extension). Can also be an Excel file (`.xlsx`, `.xls`). If the path/file doens't exist, it will be created or, if it does, will be overwritten. | `export.csv` |
| `-v`, `--verbose` | `bool` | Print more information about the data, along with a small sample (max 10 rows) before and after preprocessing. | False |
| `-h`, `--help` | &ndash; | Show help message and exit. | &ndash; |
