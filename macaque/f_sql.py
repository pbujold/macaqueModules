# -*- coding: utf-8 -*-


def correct_date(db):
    import sqlite3
    conn = sqlite3.connect(db)
    dfTable = pd.read_sql("SELECT * FROM Trials_Digital", conn)

    allDates = dfTable.date
    uniques = np.unique(allDates)
    index = {}
    for unique in uniques:
        index[tuple(unique)] = [
            i for i, x in enumerate(allDates) if x == unique
        ]

    for dateString, ii in zip(uniques, index):
        if len(dateString) != 10:
            dateString = dateString.split(sep='-')
            dateString[2] = '0' + dateString[2]
            allDates[index[ii]] = '-'.join(dateString)

    dfTable.date = allDates
    dfTable.to_sql('Trials_Digital', conn, if_exists='replace', index=False)
    conn.close()


#%%
def sort_throughFolder(pre):
    #    'C:\Users\phbuj\University Of Cambridge\OneDrive - University Of Cambridge\Lab Computer\DATA\data_Trident'

    #    pre = 'T68'

    import sqlite3
    import os
    cwd = os.getcwd()
    fileIDs = []

    for file in os.listdir(cwd):
        if file.endswith(".mat") and file.startswith(pre):
            fileIDs.append(file)

#    db = 'C:\Users\phbuj\Google Drive\Lab Data\database' + "\\" + pre
    from pathlib import Path
    home = str(Path.home())

    db = home + r'\Google Drive\Lab Data\database' + "\\" + pre
    conn = sqlite3.connect(db)

    for fileID in fileIDs:
        import_analog(fileID, conn)
    conn.close()


#%%


def import_analog(fileID, conn):
    import pandas as pd
    import numpy as np
    import datetime
    from scipy.io import loadmat

    try:
        data = loadmat(fileID, squeeze_me=True, struct_as_record=False)
    except:
        return

    cols = [
        'date', 'time', 'trialNo', 'blockNo', 'analogTime', 'eye_X', 'eye_Y',
        'Joystick_X', 'Joystick_Y'
    ]
    analogDF = pd.DataFrame(columns=cols)
    z = 0
    dfs = []

    for n in range(len(data['hh'].data.Trials)):
        reps = len(data['hh'].data.Trials[n].analog_times)

        date = data['hh'].data.Trials[n].clock[0:3]
        if len(date) == 0:
            continue

        year = str(date[0])
        month = str(date[1])
        day = str(date[2])
        if len(month) < 2:
            month = '0' + month
        if len(day) < 2:
            day = '0' + day
        time = str(data['hh'].data.Trials[n].clock[3:][0]) + ':' + str(
            data['hh'].data.Trials[n].clock[3:][1]) + ':' + str(
                data['hh'].data.Trials[n].clock[3:][2])

        if len(data['hh'].data.Trials[n].events.shape) == 1:
            if any(data['hh'].data.Trials[n].events) == 1002:
                z += 1
        elif any(data['hh'].data.Trials[n].events[:, 0] == 1002):
            z += 1

        if len(data['hh'].data.Trials[n].analog_data.shape) < 2:
            continue

        dfs.append(
            pd.DataFrame({
                'date': [year + '-' + month + '-' + day] * reps,
                'time': [time] * reps,
                'trialNo': [n] * reps,
                'blockNo': [z] * reps,
                'analogTime':
                data['hh'].data.Trials[n].analog_times.tolist(),
                'eye_X':
                data['hh'].data.Trials[n].analog_data[:, 0].tolist(),
                'eye_Y':
                data['hh'].data.Trials[n].analog_data[:, 1].tolist(),
                'Joystick_X':
                data['hh'].data.Trials[n].analog_data[:, 2].tolist(),
                'Joystick_Y':
                data['hh'].data.Trials[n].analog_data[:, 3].tolist()
            }))

    print(fileID)
    if dfs == []:
        return
    analogDF = pd.concat(dfs, ignore_index=True)
    analogDF.to_sql('Trials_Analog', conn, if_exists='append', index=False)


#%%
def csv_to_database(mCode):
    """
    create a database connection to a SQLite database
    """
    from pathlib import Path
    home = str(Path.home())
    db = home + r'\Google Drive\Lab Data\database' + "\\" + mCode

    conn = sqlite3.connect(db)
    trials = pd.read_csv('trial_TableU74_.csv')
    #    dfTable = pd.read_sql("SELECT * FROM Trials_Digital", conn)
    trials.to_sql('Trials_Digital', conn, if_exists='replace', index=False)
    conn.close()
