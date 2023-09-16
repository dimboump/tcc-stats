# This script will create the database and tables for the app
# TO BE RUN ONLY ONCE
#
# HOW TO RUN:
# $ python -m tcc_stats.create_db
import os
import pathlib
import sqlite3

import pandas as pd

historical_data = {
    2017: 1450,
    2018: 2082,
    2019: 2108,
    2020: 2863,
    2021: 3310,
    2022: 3463,
}


def get_cwd() -> pathlib.Path:
    return pathlib.Path(os.getcwd())


def main() -> int:
    cwd = get_cwd()

    try:
        db_path = pathlib.Path(cwd, 'streamlit/.streamlit/tcc_stats.db')
        if db_path.exists():
            os.remove(db_path)
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS historical_data
                  (year INTEGER PRIMARY KEY, count INTEGER)''')

        for year, count in historical_data.items():
            c.execute(
                'INSERT INTO historical_data VALUES (?, ?)', (year, count)
            )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    conn.commit()
    print(f"Database created successfully at {db_path}")

    # show the data in a dataframe
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    print(df)

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
