from __future__ import annotations

import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import s3fs
import squarify
import streamlit as st
from matplotlib import pyplot as plt

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

fs = s3fs.S3FileSystem(anon=False)

BUCKET_NAME = 'tcc-stats'
DATA_FILE = 'tcc_stats.csv'


# Basic authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state['password'] == st.secrets['password']:
            st.session_state['password_correct'] = True
            st.session_state['password'] = None  # don't store password
        else:
            st.session_state['password_correct'] = False

    if 'password_correct' not in st.session_state:
        # First run, show input for password
        st.text_input(
            'Password', type='password',
            on_change=password_entered, key='password'
        )
        return False
    elif not st.session_state['password_correct']:
        # Password incorrect, show input + error
        st.text_input(
            'Password', type='password',
            on_change=password_entered, key='password'
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True


if not check_password():
    st.stop()


def load_historical_data() -> pd.DataFrame:
    with fs.open(f'{BUCKET_NAME}/{DATA_FILE}', 'r', encoding='utf-8') as f:
        historical_data = pd.read_csv(f, index_col=0, header=0)
    return historical_data


def save_data(new_data: pd.DataFrame) -> None:
    with fs.open(f'{BUCKET_NAME}/{DATA_FILE}', 'w', encoding='utf-8') as f:
        new_data.to_csv(f, index=True, header=True)


def update_selected_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Update the historical data in the database with the selected year.
    If the year is already in the database, just update the number of requests.
    Otherwise, add the year and the number of requests to the database."""
    df = df.reset_index()
    data = load_historical_data()

    if year not in data.index:
        new_data = pd.DataFrame({'count': [len(df)]}, index=[year])
        data = pd.concat([data, new_data])
        data = data.sort_index()
    else:
        data.loc[year, 'count'] = len(df)

    save_data(data)

    return data


def remove_selected_year(historical_data: pd.DataFrame, year: int) -> None:
    data = historical_data.drop(year, axis=0)
    with fs.open(f"{BUCKET_NAME}/{DATA_FILE}", 'w', encoding='utf-8') as f:
        data.to_csv(f, index=True, header=True)
    st.success(f'Data for **{year}** were deleted from the database.')


def app():
    st.title(C.TITLE)

    st.header(':blue[Step 1️⃣:] Get single file', anchor='step1')
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

        testing = st.checkbox('_:orange[testing mode]_ 🧪', value=False)

        with st.expander("**:blue[What's new (v2023.10.07)]**",
                         expanded=True):
            st.markdown("""\
                - Empty sheets when uploading `xlsx` files are now ignored.
                - Added option to mark trainees, along with a respective plot.
                - `CDC` is now included in the list of TCCers.
                - Bug fixes and performance improvements.
                - Some UI improvements; icons in the sidebar.
                """)

        data_header = st.empty()
        data_header.header('📉 **Historical data**', divider='gray')

        history = st.empty()

        historical_data = load_historical_data()
        history.dataframe(
            historical_data,
            use_container_width=True,
            column_config={'_index': st.column_config.NumberColumn(format='%d')}
        )

        st.header('⚙️ **Global options**', divider='gray')

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
        mark_trainees = st.checkbox('**Mark trainees**', value=False)
        with_trainees = not mark_trainees
        tccers = st.text_input('TCCers (staff):', value=', '.join(C.TCCERS),
                               disabled=with_trainees)
        TCCERS = [tccer.strip() for tccer in tccers.split(',')]

    with rcol1_1:
        testing_mode = st.empty()
        if testing:
            testing_mode.button(f'Remove {year} from database',
                                on_click=remove_selected_year,
                                args=(historical_data, year))

    with rcol1:
        if not files:
            st.info('Upload the :orange[`XLSX`] file for the whole year or the '
                    ':orange[`CSV`] files for each month separately.')
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
                        if not df_temp.empty:
                            uploaded_months.add(i)
                if uploaded_months != set(range(1, 13)):
                    msg1.warning('**Not all sheets were provided!**')
                    msg1.warning('**Statistics were extracted only for the '
                                 'provided sheets.**')
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
                st.balloons()
                n_files = len(files)
            else:
                n_files = len(files) - n_files
            s = '' if n_files == 1 else 's'
            so_far = ''

            if year == current_year and n_files < 12:
                so_far = ' :red[so far]'
                latest_month = C.MONTHS[max(uploaded_months)-1]
                filename = f'{year}_until_{latest_month}.{ext}'
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

            updated_data = update_selected_year(df, year)
            data_header.header('📈 **Updated data**', divider='gray')
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

    st.header(':blue[Step 2️⃣:] Exploratory Data Analysis', anchor='step2')

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
        return f'{x:.1f}%\n({total * x / 100:.0f})'

    with lcol2_1:
        st.subheader('')
        st.subheader('Basic statistics')
        st.subheader('')

    with ccol2_2:
        # plot the number of requests since 2017
        st.markdown('#### Number of requests per year')
        st.subheader('')

        with st.expander('**Options**', expanded=True):
            show_diff = st.checkbox('**Chiara feature (show % differences)**',
                                    value=True)
            hide_yticks = st.checkbox('**Hide y-axis ticks**', value=False)

        fig, ax = plt.subplots()
        ax.plot(updated_data.keys(), updated_data.values(),
                color='#0173b2', linewidth=2, zorder=2)
        for i, (x, y) in enumerate(updated_data.items()):
            if show_diff and i > 0:
                counts = list(updated_data.values())
                diff = y - counts[i-1]
                diff_perc = round(diff / counts[i-1] * 100, 1)
                color = 'green' if diff > 0 else 'red'
                curr_requests = counts[i]
                ax.annotate(f'{diff_perc:+}%', (x, y), ha='center', va='bottom',
                            xytext=(x-0.1, curr_requests+100), size=10,
                            color=color, weight='bold', zorder=10, alpha=0.5)
            size = 18 if x == year else 14
            color = '#de8f05' if x == year else 'gray'
            ha = 'left'
            y_offset = 150
            ax.annotate(f'{y:,}', (x, y), xytext=(x+0.05, y-y_offset), ha=ha,
                        va='bottom', size=size, color=color, weight='bold')
        data_min = min(updated_data.keys())
        data_max = max(updated_data.keys())
        ax.set_xlim(data_min - 1, data_max + 1)
        ax.set_xticks(range(data_min, data_max + 1))
        ax.set_ylim(round(min(updated_data.values()) * 0.9, -1),
                    round(max(updated_data.values()) * 1.1, -2))
        if hide_yticks:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
        ax.set_title(f'Number of requests ({data_min}-{data_max})',
                     size=16, weight='bold')
        st.pyplot(fig, clear_figure=True)

    st.divider()

    st.header(':blue[Step 3️⃣:] Plots', anchor='step3')
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

    with st.expander('**Options**', expanded=True):
        plot_type = st.selectbox('Select plot type',
                                 options=['Bar', 'Line'], index=0)

    fig, ax = plt.subplots()

    plot_type = plot_type.lower()
    df_imp_month.plot(kind=plot_type, ax=ax, color=['#0173b2', '#de8f05'])
    if plot_type == 'line':
        # prevent xtick labels from hiding from the x-axis
        ax.set_xticks(range(len(df_imp_month.index)))
        ax.set_xticklabels(df_imp_month.index[::1])  # https://saturncloud.io/blog/how-to-show-more-ticks-on-the-xaxis-in-pandas-plot/
        for i in ax.lines:
            i.set_marker('o')

    max_value = max(max(df_imp_month[True]), max(df_imp_month[False]))
    ax.set_ylim(0, round(max_value * 1.1, -2))
    # give some space on the left and the right of the plot
    # ax.set_xlim(df_imp_month.index[0] - pd.Timedelta(days=1),
    #             df_imp_month.index[-1] + pd.Timedelta(days=1))
    ax.set_title(f'Improved vs OK per month ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')

    # show the values on top of the bars where their color matches the bar's
    if plot_type == 'bar':
        for p in ax.patches:
            color = '#0173b2' if p.get_facecolor() == rgba_value else '#de8f05'
            annotation_text = f'{p.get_height():,}'
            annotation_xy = (p.get_x() + p.get_width() / 2, p.get_height())
            ax.annotate(annotation_text, annotation_xy,
                        ha='center', va='center', size=12, color=color,
                        xytext=(0, 6), textcoords='offset points')
    elif plot_type == 'line':
        for i, p in enumerate(ax.lines):
            # plot a line between the y values of each x value
            for x, y in zip(p.get_xdata(), p.get_ydata()):
                ax.annotate(f'{y:,}', (x, y), ha='center', va='bottom',
                            xytext=(0, 6), textcoords='offset points',
                            size=12, color=p.get_color())

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

    plt.xlabel('')
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
    req_labels = [f'{dg_code}\n{count:,}' for dg_code, count
                  in zip(df_dg['dg_code'], df_dg['counts'])]
    squarify.plot(sizes=df_dg['counts'], label=req_labels,
                  color=mcolors.TABLEAU_COLORS,
                  alpha=0.8, ax=ax, text_kwargs={'size': 6, 'color': 'white'})
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
    df_dg_imp = df_dg_imp[df_dg_imp['total'] >= 10]
    df_dg_imp = df_dg_imp.reindex(df_dg_imp.index.tolist() + ['Other'],
                                  fill_value=0, method=None)
    df_dg_imp.loc['Other', True] = df_dg_imp_other[True].sum(axis=0)
    df_dg_imp.loc['Other', False] = df_dg_imp_other[False].sum(axis=0)

    # drop the 'total' column
    df_dg_imp = df_dg_imp.drop(columns=['total'])

    # reindex the dataframe to have all available dg_codes
    dg_codes = list(sorted(df_dg_imp.index.tolist()))
    df_dg_imp = df_dg_imp.reindex(dg_codes, fill_value=0, method=None)

    with st.expander('**Options**', expanded=True):
        alphabetical = st.checkbox('Sort alphabetically', value=False)
        if alphabetical:
            df_dg_imp = df_dg_imp.sort_index()
        else:
            # sort the columns by the total number of requests
            df_dg_imp = df_dg_imp.sort_values(True, ascending=False)

    fig, ax = plt.subplots()
    df_dg_imp.plot(kind='bar', ax=ax, color=['#0173b2', '#de8f05'])

    # set ylim to the max OK *or* Improved value rounded to the nearest hundred
    # if max_value >= 1,000, otherwise rounded to the nearest thousand
    max_ok_imp = max(max(df_dg_imp[False]) * 1.1, max(df_dg_imp[True]) * 1.1)
    round_by = -1 if max_ok_imp >= 100 else -2
    ax.set_ylim(0, round(max_ok_imp, round_by))

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
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='center', size=12, color=color,
                    xytext=(0, 6), textcoords='offset points')

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

    plt.xlabel('')
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # plot improved per doc_type
    st.markdown('#### Status per Document Type')
    st.subheader('')

    df_dt_imp = df.groupby(['doc_type', 'improved']).size().unstack()

    # create a new column with the total count for each doctype
    df_dt_imp['total'] = df_dt_imp.sum(axis=1)

    # find the doctypes with less than 5 requests and group them into 'Other'
    df_dt_imp_other = df_dt_imp[df_dt_imp['total'] < 5]
    df_dt_imp = df_dt_imp[df_dt_imp['total'] >= 5]
    df_dt_imp = df_dt_imp.reindex(df_dt_imp.index.tolist() + ['Other'],
                                  fill_value=0, method=None)
    df_dt_imp.loc['Other', True] = df_dt_imp_other[True].sum(axis=0)
    df_dt_imp.loc['Other', False] = df_dt_imp_other[False].sum(axis=0)

    # drop the 'total' column
    df_dt_imp = df_dt_imp.drop(columns=['total'])

    # reindex the dataframe to have all available doc_types
    doc_types = list(sorted(df_dt_imp.index.tolist()))
    df_dt_imp = df_dt_imp.reindex(doc_types, fill_value=0, method=None)

    with st.expander('**Options**', expanded=True):
        if alphabetical := st.checkbox('Sort alphabetically',
                                       value=False, key='doctypes_alpha'):  # noqa
            df_dt_imp = df_dt_imp.sort_index()
        else:
            # sort the columns by the total number of requests
            df_dt_imp = df_dt_imp.sort_values(True, ascending=False)

    fig, ax = plt.subplots()
    df_dt_imp.plot(kind='bar', ax=ax, color=['#0173b2', '#de8f05'])

    # set ylim to the max OK *or* Improved value rounded to the nearest hundred
    # if max_value >= 1,000, otherwise rounded to the nearest thousand
    max_ok_imp = max(max(df_dt_imp[False]) * 1.1, max(df_dt_imp[True]) * 1.1)
    round_by = -1 if max_ok_imp >= 100 else -2
    ax.set_ylim(0, round(max_ok_imp, round_by))

    ax.set_title(f'Status per document type ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')

    # show the values on top of the bars where their color matches the bar's
    for p in ax.patches:
        color = '#0173b2' if p.get_facecolor() == rgba_value else '#de8f05'
        ax.annotate(f'{p.get_height():,}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='center', size=12, color=color,
                    xytext=(0, 6), textcoords='offset points')

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

    plt.xlabel('')
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)

    st.divider()

    st.markdown('#### Time needed')
    st.subheader('')

    # we need to convert the 'minutes' column to a string
    # before converting it into the time segments
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
    max_counts = max(df_time['counts']) * 1.1
    round_by = -2 if max_counts >= 1000 else -3
    ax.set_ylim(0, round(max_counts, round_by))
    ax.set_title(f'Time needed ({year})', size=16,
                 weight='bold', pad=60)
    ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
            transform=ax.transAxes, size=14, ha='center')
    # show the values on top of the bars
    for p in ax.patches:
        color = '#0173b2'
        ax.annotate(f'{p.get_height():,}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='center', size=12, color=color,
                    xytext=(0, 6), textcoords='offset points')

    plt.xlabel('')
    plt.xticks(size=14, rotation=45, ha='right')
    plt.yticks(size=14)
    ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
    ax.get_legend().remove()
    fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
    st.pyplot(fig, clear_figure=True)

    if mark_trainees:
        st.divider()

        st.markdown("#### Trainees' Contribution")
        st.subheader('')

        df_trainees = df.groupby(['month', 'trainee']).size().unstack()
        df_trainees = df_trainees.fillna(0)
        df_trainees = df_trainees.astype(int)
        df_trainees = df_trainees.reindex(months, fill_value=0, method=None)

        # plot the bar chart to show the number of trainees vs staff
        # per month
        fig, ax = plt.subplots()
        df_trainees.plot(kind='bar', ax=ax, color=['#0173b2', '#de8f05'])
        max_trainees = max(df_trainees[True] * 1.1)
        max_staff = max(df_trainees[False] * 1.1)
        round_by = -1 if max(max_trainees, max_staff) >= 100 else -2
        ax.set_ylim(0, round(max(max_trainees, max_staff), round_by))
        ax.set_title(f"Trainees' Contribution ({year})", size=16,
                     weight='bold', pad=60)
        ax.text(0.5, 1.125, f'Total documents checked: {total:,}',
                transform=ax.transAxes, size=14, ha='center')
        # show the values on top of the bars
        for p in ax.patches:
            color = '#0173b2' if p.get_facecolor() == rgba_value else '#de8f05'
            ax.annotate(f'{p.get_height():,}',
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='center', size=12, color=color,
                        xytext=(0, 6), textcoords='offset points')

        plt.xlabel('')
        plt.xticks(size=14, rotation=45, ha='right')
        plt.yticks(size=14)

        ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks()])
        trainees_counts = df_trainees[True].sum()
        staff_counts = df_trainees[False].sum()
        total = trainees_counts + staff_counts
        trainees_perc = round(trainees_counts / total * 100, 1)
        staff_perc = round(staff_counts / total * 100, 1)
        _labels = [f'Trainees - {trainees_counts:,} ({trainees_perc}%)',
                   f'Staff - {staff_counts:,} ({staff_perc}%)']
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  labels=_labels, ncol=len(_labels),
                  prop={'weight': 'bold', 'size': 14})

        fig.set_size_inches(len(uploaded_months) * 1.5, 6)  # no overlapping labels
        st.pyplot(fig, clear_figure=True)


if __name__ == '__main__':
    app()
