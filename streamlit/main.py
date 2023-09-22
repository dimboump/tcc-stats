from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from sqlite3 import Connection as Conn

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import squarify
from matplotlib import pyplot as plt

import streamlit as st
from tcc_stats import constants as C
from tcc_stats.commands.extract import preprocess_data

st.set_page_config(
    page_title='TCC Stats',
    page_icon=':bar_chart:',
    layout='wide',
    initial_sidebar_state='auto'
)

plt.style.use('seaborn-v0_8')
mpl.rcParams['figure.dpi'] = 180

current_year = datetime.now().year

rgba_value = (0.00392156862745098, 0.45098039215686275, 0.6980392156862745, 1.0)


def get_conn() -> Conn:
    try:
        conn = sqlite3.connect('../streamlit/.streamlit/tcc_stats.db')
    except sqlite3.OperationalError:
        st.error('Error: Could not connect to the database.')
        st.stop()
    return conn


def get_tcc_stats(conn: Conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    return df.set_index('year')


def update_selected_year(
    conn: Conn,
    df: pd.DataFrame,
    year: int
) -> pd.DataFrame:
    """Update the historical data in the database with the selected year.
    If the year is already in the database, just update the number of requests.
    Otherwise, add the year and the number of requests to the database."""
    df = df.reset_index()

    c = conn.cursor()
    c.execute('SELECT * FROM historical_data WHERE year=?', (year,))
    if c.fetchone() is None:
        c.execute('INSERT INTO historical_data VALUES (?, ?)', (year, len(df)))
    else:
        c.execute('UPDATE historical_data SET count=? WHERE year=?',
                  (len(df), year))
    conn.commit()

    return get_tcc_stats(conn)


def app():
    st.title(C.TITLE)

    st.header(':blue[Step 1:] Get single file', anchor='step1')
    st.subheader('')

    lcol1, rcol1 = st.columns([0.4, 0.6], gap='medium')
    lcol1_1, rcol1_1 = lcol1.columns([0.55, 0.45], gap='medium')

    with lcol1_1:
        year = st.selectbox('Year:', label_visibility='collapsed',
                            key='year', options=reversed(C.YEARS))

    with lcol1:
        files = st.file_uploader('Upload files:', type=['xlsx', 'csv'],
                                 accept_multiple_files=True, key='files',
                                 label_visibility='collapsed')
        msg1, msg2, msg3 = st.empty(), st.empty(), st.empty()

    # sidebar
    with st.sidebar:

        testing = st.checkbox('_:orange[testing mode]_', value=False)

        data_header = st.empty()
        data_header.header('**Historical data**', divider='gray')

        history = st.empty()

        conn = get_conn()
        historical_data = get_tcc_stats(conn)
        history.dataframe(
            historical_data,
            use_container_width=True,
            column_config={'_index': st.column_config.NumberColumn(format='%d')}
        )

        st.header('**Options**', divider='gray')

        st.subheader('**:blue[Output file]**')

        # Customize the extension of the resulting file
        ext = st.selectbox('**File extension**',
                           options=['xlsx', 'csv'], key='ext', index=0)

        st.subheader('**:blue[Data manipulation]**')

        # Exclude confidential requests
        confidentials = st.checkbox('**Include confidentials**', value=True)
        if not confidentials:
            st.warning('**Note:** Excluding confidentials will also affect '
                       'the number of requests stored in the database.')
        st.markdown('<hr style="margin: 5px 0 15px;">', unsafe_allow_html=True)

        # Distinguish staff and trainees
        mark_trainees = st.checkbox('**Staff vs Trainees**', value=False)
        with_trainees = not mark_trainees
        tccers = st.text_input('TCCers (staff):', value=', '.join(C.TCCERS),
                               disabled=with_trainees)
        TCCERS = [tccer.strip() for tccer in tccers.split(',')]

        st.subheader('**:blue[Data visualization]**')

        show_diff = st.checkbox('**Show difference with previous year**',
                                value=False)

    with rcol1_1:

        def remove_selected_year() -> None:
            conn = get_conn()
            c = conn.cursor()
            c.execute('DELETE FROM historical_data WHERE year=?', (year,))
            conn.commit()
            st.success(f'Data for **{year}** were deleted from the database.')

        testing_mode = st.empty()
        if testing:
            testing_mode.button(f'Remove {year} from database',
                                on_click=remove_selected_year)

    with rcol1:
        if not files:
            st.info('Upload the :orange[`xlsx`] file for the whole year or the '
                    ':orange[`csv`] files for each month separately.')
        else:
            C.STATE['files'] = files
            block_download = False
            n_files = len(files)
            filenames = {file.name for file in files}
            filenames, fileexts = zip(*[os.path.splitext(file)
                                        for file in filenames])
            fileexts = set(fileexts)
            if all([ext == '.xlsx' for ext in fileexts]):
                if n_files > 1:
                    msg1.error('Cannot process more than one xlsx file.')
                    st.stop()
                msg1.info('Separate sheets were provided.')
                msg1.info('Checking if the sheets are named correctly...')
                df = pd.DataFrame()
                uploaded_months = set()
                for file in files:
                    df_sheets = pd.read_excel(file, usecols='A:I', names=C.COLS,
                                              sheet_name=None, index_col=None)
                    for i, month_name in enumerate(C.MONTHS, start=1):
                        sheetname = f"ANTE - {month_name.upper()} {year}"
                        if isinstance(df_sheets, dict):
                            df_temp = df_sheets[sheetname]
                        # add the year and the month to the dataframe
                        df_temp = df_temp.assign(year=year, month=C.MONTHS[i-1])
                        df = pd.concat([df, df_temp], ignore_index=True)
                        uploaded_months.add(i)
                if uploaded_months != set(range(1, 13)):
                    msg1.warning('**Not all sheets are named correctly!**')
                    msg1.warning('**Please rename the sheets and try again.**')
                    st.stop()
                msg1.success('All months were provided as sheets.')
            elif n_files > 1 and all([ext == '.csv' for ext in fileexts]):
                msg1.info('Separate files were provided.')
                uploaded_months = {int(file.name.split('.')[0])
                                   for file in files}
                dfs = [
                    (
                        pd.read_csv(file, index_col=None,
                                    names=C.COLS, sep=';')
                        .assign(
                            year=pd.to_datetime(year, format='%Y').year,
                            #  assign the month with its name
                            month=C.MONTHS[int(file.name.split('.')[0]) - 1],
                        )
                    )
                    for file in files
                ]
                df = pd.concat(dfs, ignore_index=True)
            else:
                msg1.error('Cannot process both xlsx and csv files at once.')
                st.stop()
            if n_files == 12 and uploaded_months == set(range(1, 13)):
                msg1.success('All months were provided.')
            elif n_files == 12 and uploaded_months != set(range(1, 13)):
                msg1.warning('**12 files were provided, but not all of them '
                             'are named correctly!**')
            elif n_files > 12:
                msg1.error("""**More than 12 files were provided!
                           To continue, remove the unnecessary files.**""")
                block_download = True
                st.stop()

            data_files = []
            for file in files:
                data_files.append(file)
                n_files -= 1
            if n_files == 0:
                msg2.success('All files were loaded successfully.')
                n_files = len(files)
            else:
                n_files = len(files) - n_files
            s = '' if n_files == 1 else 's'
            so_far = ''

            if year == current_year and n_files < 12:
                so_far = ' :red[so far]'
                filename = f'{year}_{n_files}_month{s}.{ext}'
            else:
                filename = f'{year}_all.{ext}'

            df = preprocess_data(df, confidentials=confidentials,
                                 mark_trainees=mark_trainees, staff=TCCERS)
            if ext == 'xlsx':
                df.to_excel(filename, index=False, freeze_panes=(1, 1))
            elif ext == 'csv':
                df.to_csv(filename, index=False, sep=';')
            with open(filename, 'rb') as f:
                stats = f.read()
            msg2.info(f'Loaded **`{len(files)}`** file{s} with **`{len(df)}`** '
                      'rows of data.')
            if len(files) != n_files and n_files != 0:
                msg3.warning(f'Duplicate files found! {len(files) - n_files} '
                             f'file{s} were ignored.')
            st.success(f'Extracted the statistics for **{year}**{so_far}.')

            updated_data = update_selected_year(conn, df, year)
            data_header.header('**Updated data**', divider='gray')
            history.dataframe(
                updated_data,
                use_container_width=True,
                column_config={
                    '_index': st.column_config.NumberColumn(format='%d'),
                }
            )

            updated_data = updated_data.to_dict()['count']
            print("Updated data dict:", updated_data)

            preview = st.empty()
            preview.dataframe(
                df.sample(10), use_container_width=True,
                column_config={
                    'year': st.column_config.NumberColumn(format='%d'),
                }
            )

            def refresh_sample() -> None:
                preview.dataframe(
                    df.sample(10), use_container_width=True,
                    column_config={
                        'year': st.column_config.NumberColumn(format='%d'),
                    }
                )

            help = 'Please upload all 12 files first.' if block_download else ''
            st.download_button(f'Download {filename}', type='primary',
                               file_name=filename, data=stats,
                               disabled=block_download, help=help)
            st.button('Refresh sample', on_click=refresh_sample,
                      help="Get another 10 random rows from the data.")

    st.divider()

    st.header(':blue[Step 2.] Exploratory Data Analysis', anchor='step2')

    lcol2_1, rcol2_1 = st.columns(2, gap='medium')
    _, ccol2_2, _ = st.columns([0.2, 0.6, 0.2], gap='small')
    stat_col1, stat_col2 = lcol2_1.columns(2, gap='small')
    stat_col3, stat_col4 = rcol2_1.columns(2, gap='small')

    if not files:
        st.info('EDA will be performed once the data in Step 1 are loaded.')
        st.stop()

    st.subheader('')

    with stat_col1:
        st.metric('Year', year)
    with stat_col2:
        st.metric('Ex-Ante Requests', f'{len(df):,}')
    with stat_col3:
        st.metric('Avg Requests per working day', round(len(df) / 260, 2))
    with stat_col4:
        last_year = year - 1
        last_year_diff = len(df) - updated_data[last_year]
        st.metric(f'Difference from {last_year}',
                  f'{round(last_year_diff / len(df) * 100, 2)}%',
                  delta=f'{len(df) - updated_data[last_year]:,}')

    total = len(df)

    def pie_fmt(x):
        return '{:.1f}%\n({:.0f})'.format(x, total * x / 100)

    with lcol2_1:
        st.subheader('')
        st.subheader('Basic statistics')
        st.subheader('')

    with ccol2_2:
        # plot the number of requests since 2017
        st.markdown('#### Number of requests per year')
        st.subheader('')

        fig, ax = plt.subplots()
        ax.plot(updated_data.keys(), updated_data.values(),
                color='#0173b2', linewidth=2, zorder=2)
        for i, (x, y) in enumerate(updated_data.items()):
            if show_diff and i > 0:
                diff = y - list(updated_data.values())[i-1]
                color = 'green' if diff > 0 else 'red'
                curr_requests = list(updated_data.values())[i]
                ax.annotate(f'{diff:+}', (x, y), ha='center', va='bottom',
                            xytext=(x-0.1, curr_requests+100), size=10,
                            color=color, weight='bold', zorder=10, alpha=0.5)
            size = 18 if x == year else 14
            color = '#de8f05' if x == year else 'gray'
            ha = 'left'
            y_offset = 150
            ax.annotate(f'{y:,}', (x, y), xytext=(x+0.05, y-y_offset), ha=ha,
                        va='bottom', size=size, color=color, weight='bold')
        ax.set_ylim(round(min(updated_data.values()) - 250, 1),
                    round(max(updated_data.values()) + 250, -2))
        ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
        ax.set_title(f'Number of requests ({min(updated_data.keys())}-'
                     f'{max(updated_data.keys())})', size=16, weight='bold')
        st.pyplot(fig, clear_figure=True)

    st.divider()

    st.header(':blue[Step 3.] Plots', anchor='step3')
    st.subheader('')

    lcol3_1, ccol3_1, rcol3_1 = st.columns(3, gap='large')

    with lcol3_1:
        # plot pie chart of improved vs OK
        st.markdown('<h4 style="text-align: center;">Improved vs OK</h4>',
                    unsafe_allow_html=True)
        st.subheader('')

        df_result = df['improved'].value_counts()
        fig, ax = plt.subplots()
        improve_labels = ['Improved' if i else 'OK' for i in df_result.index]
        ax.pie(df_result, labels=improve_labels, autopct=pie_fmt, startangle=90,
               colors=['#0173b2', '#de8f05'], textprops={'color': 'white'})
        ax.set_title(f'Improved vs OK ({year})', size=16,
                     weight='bold')
        plt.legend(loc='upper right')
        st.pyplot(fig, clear_figure=True)

    with ccol3_1:
        # plot pie chart of Confidential vs Non-Confidential
        st.markdown('<h4 style="text-align: center;">Confidentials</h4>',
                    unsafe_allow_html=True)
        st.subheader('')

        # create a new temporary column to get the confidentials
        df['confidential'] = df['requester_code'].str.match('Confidential')
        df_result = df['confidential'].value_counts()
        fig, ax = plt.subplots()
        conf_labels = ['Confidential' if i else 'Non-Confidential'
                       for i in df_result.index]
        ax.pie(df_result, labels=conf_labels, autopct=pie_fmt, startangle=90,
               colors=['#0173b2', '#de8f05'], textprops={'color': 'white'})
        ax.set_title(f'Confidential Requests ({year})', size=16,
                     weight='bold')
        plt.legend(loc='upper right')
        st.pyplot(fig, clear_figure=True)
        df = df.drop('confidential', axis=1)

    with rcol3_1:
        # plot pie chart of PDFs
        st.markdown('<h4 style="text-align: center;">PDFs vs Non-PDFs</h4>',
                    unsafe_allow_html=True)
        st.subheader('')

        df_result = df['pdf'].value_counts()
        fig, ax = plt.subplots()
        pdf_labels = ['PDF' if i else 'Other' for i in df_result.index]
        ax.pie(df_result, labels=pdf_labels, autopct=pie_fmt, startangle=90,
               colors=['#0173b2', '#de8f05'], textprops={'color': 'white'})
        ax.set_title(f'PDFs vs Non-PDFs ({year})', size=16,
                     weight='bold')
        plt.legend(loc='upper right')
        st.pyplot(fig, clear_figure=True)

    st.divider()

    st.markdown('#### Improved vs OK per month')
    st.subheader('')

    df_imp_month = df.groupby(['month', 'improved']).size().unstack()
    df_imp_month = df_imp_month.fillna(0)
    df_imp_month = df_imp_month.astype(int)

    # reindex the dataframe to have all available months
    months = list(sorted(df_imp_month.index.tolist(), key=C.MONTHS.index))
    df_imp_month = df_imp_month.reindex(months, fill_value=0, method=None)

    fig, ax = plt.subplots()
    df_imp_month.plot(kind='bar', ax=ax, color=['#0173b2', '#de8f05'])
    ax.set_ylim(0, round(max(df_imp_month.sum(axis=1)), -2))
    ax.set_title(f'Improved vs OK per month ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')
    # show the values on top of the bars where their color matches the bar's
    for p in ax.patches:
        color = '#0173b2' if p.get_facecolor() == rgba_value else '#de8f05'
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2.,
                                            p.get_height()),
                    ha='center', va='center', size=14, color=color,
                    xytext=(0, 10), textcoords='offset points')
    # Labels based on `improve_labels` but with the total number
    # of requests for the year along with the percentage, e.g.:
    # ['Improved (1,234) (12.3%)', 'OK (4,567) (45.6%)']
    improved_counts = len(df[df['improved'].eq(True)])
    ok_counts = len(df[df['improved'].eq(False)])
    total = improved_counts + ok_counts
    improved_perc = round(improved_counts / total * 100, 1)
    ok_perc = round(ok_counts / total * 100, 1)
    _labels = [f'OK - {ok_counts:,} ({ok_perc}%)',
               f'Improved - {improved_counts:,} ({improved_perc}%)']
    ax.legend(loc='upper center', labels=_labels, ncol=len(_labels),
              bbox_to_anchor=(0.5, 1.1), prop={'weight': 'bold', 'size': 14})
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)

    st.divider()

    st.markdown('#### Requestors')
    st.subheader('')

    df_dg = df.groupby('dg_code').size().reset_index(name='counts')
    df_dg = df_dg.sort_values('counts', ascending=False)
    df_dg = df_dg.reset_index(drop=True)

    # find the DGs with less than 10 requests and group them into 'Other'
    df_dg_other = df_dg[df_dg['counts'] < 10]
    df_dg_other_count = df_dg_other['counts'].sum()
    df_dg = df_dg[df_dg['counts'] >= 10]
    df_dg = pd.concat([df_dg, pd.DataFrame(
        [['Other', df_dg_other_count]],
        columns=['dg_code', 'counts']
    )])

    df_dg = df_dg[df_dg['dg_code'] != 'nan']

    # plot the treemap
    fig, ax = plt.subplots()
    squarify.plot(sizes=df_dg['counts'], label=df_dg['dg_code'],
                  color=mcolors.TABLEAU_COLORS,
                  alpha=0.8, ax=ax, text_kwargs={'size': 6})
    ax.set_title(f'Requestors ({year})', size=16, weight='bold')
    ax.axis('off')
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # plot improved per dg_code
    st.markdown('#### Status per Requestor')
    st.subheader('')

    df_dg_imp = df.groupby(['dg_code', 'improved']).size().unstack()

    # create a new column with the total count for each requestor
    df_dg_imp['total'] = df_dg_imp.sum(axis=1)

    # find the requestors with less than 10 requests and group them into 'Other'
    df_dg_imp_other = df_dg_imp[df_dg_imp['total'] < 10]
    df_dg_imp_other_true = df_dg_imp_other[True].sum(axis=0)
    df_dg_imp_other_false = df_dg_imp_other[False].sum(axis=0)
    df_dg_imp = df_dg_imp[df_dg_imp['total'] >= 10]
    df_dg_imp = df_dg_imp.reindex(df_dg_imp.index.tolist() + ['Other'],
                                  fill_value=0, method=None)
    df_dg_imp.loc['Other', True] = df_dg_imp_other_true
    df_dg_imp.loc['Other', False] = df_dg_imp_other_false

    # drop the 'total' column
    df_dg_imp = df_dg_imp.drop(columns=['total'])

    # reindex the dataframe to have all available dg_codes
    dg_codes = list(sorted(df_dg_imp.index.tolist()))
    df_dg_imp = df_dg_imp.reindex(dg_codes, fill_value=0, method=None)

    with st.expander('**Options**'):
        if alphabetical := st.checkbox('Sort alphabetically', value=False):  # noqa
            df_dg_imp = df_dg_imp.sort_index()
        else:
            # sort the columns by the total number of requests
            df_dg_imp = df_dg_imp.sort_values(True, ascending=False)

    fig, ax = plt.subplots()
    df_dg_imp.plot(kind='bar', ax=ax, color=['#0173b2', '#de8f05'])
    ax.set_ylim(0, round(max(df_dg_imp.sum(axis=1)), -2))
    ax.set_title(f'Status per Requestor ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')
    # show the values on top of the bars where their color matches the bar's
    for p in ax.patches:
        color = '#0173b2' if p.get_facecolor() == rgba_value else '#de8f05'
        try:
            requestor_total = \
                df_dg_imp[df_dg_imp.index == p.get_x()].sum(axis=1).values[0]
            requestor_perc = f'\n{p.get_height() / requestor_total * 100:.1f}%'
        except IndexError:
            requestor_total = 0
            requestor_perc = ''
        ax.annotate(f'{p.get_height():,}{requestor_perc}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', size=14, color=color,
                    xytext=(0, 10), textcoords='offset points')
    # Labels based on `improve_labels` but with the total number
    # of requests for the year along with the percentage, e.g.:
    # ['Improved (1,234) (12.3%)', 'OK (4,567) (45.6%)']
    improved_counts = len(df[df['improved'].eq(True)])
    ok_counts = len(df[df['improved'].eq(False)])
    total = improved_counts + ok_counts
    improved_perc = round(improved_counts / total * 100, 1)
    ok_perc = round(ok_counts / total * 100, 1)
    _labels = [f'OK - {ok_counts:,} ({ok_perc}%)',
               f'Improved - {improved_counts:,} ({improved_perc}%)']
    ax.legend(loc='upper center', labels=_labels, ncol=len(_labels),
              bbox_to_anchor=(0.5, 1.1), prop={'weight': 'bold', 'size': 14})
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)

    st.divider()

    st.markdown('#### Time needed')
    st.subheader('')

    # we need to convert the 'minutes' column to a string and then convert it
    # into the time segments
    df_time = df.copy()
    df['minutes'] = df['minutes'].astype(int)
    df_time = df_time.assign(
        duration=pd.cut(
            df['minutes'],
            bins=[0, 15, 30, 60, 90, 120, 180,
                  240, 300, 360, 420, np.inf],
            labels=['0-15 min', '15-30 min', '30-60 min', '1-1.5 hours',
                    '1.5-2 hours', '2-3 hours', '3-4 hours', '4-5 hours',
                    '5-6 hours', '6-7 hours', '7+ hours']
        )
    )

    # group by the time segments and count the number of requests
    df_time = df_time.groupby('duration').size().reset_index(name='counts')
    df_time = df_time.sort_values('duration', ascending=True)
    df_time = df_time.set_index('duration')

    # plot the bar chart
    fig, ax = plt.subplots()
    df_time.plot(kind='bar', ax=ax, color=['#0173b2'])
    ax.set_ylim(0, round(max(df_time['counts']), -3))
    ax.set_title(f'Time needed ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')
    # show the values on top of the bars
    for p in ax.patches:
        color = '#0173b2'
        try:
            requestor_total = \
                df_time[df_time.index == p.get_x()].sum(axis=1).values[0]
        except IndexError:
            requestor_total = 0
        ax.annotate(f'{p.get_height():,}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', size=14, color=color,
                    xytext=(0, 10), textcoords='offset points')
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    ax.get_legend().remove()
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)


if __name__ == '__main__':
    app()
