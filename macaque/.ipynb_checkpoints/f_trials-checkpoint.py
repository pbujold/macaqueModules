"""
Module of functions that apply to/get individual trial data for behavioural sessions (monkeys).

Functions
----------
load_monkey : load_monkey(mCode)
    returns: *trials, dates*
get_short : get_short(trials)
    **
del_errorType : del_errorTrials(trials, errorType = 'all')
    .
: by_sequenceType(trials, sequenceType)
    .
: by_sequenceMode(trials, sequenceMode)
    .
: by_date(df_full, date, sequence_mode = 'all', sequence_type = 'all', filter_errors = True)
    .
by_range :
    .
: by_gamble(df_full)
    .
"""

import numpy as np
import matplotlib as pyplot
import pandas as pd
from macaque.f_toolbox import *
import re
tqdm = ipynb_tqdm()


#%%
def load_trialsCSV(mCode=[], shorten=False):
    '''
    load_monkey returns a dataframe from the csv trial file and a numpy array of all session dates.

    Parameters
    ----------
    mCode : string
        'T73' is Tigger \n
        'U74' is Ugo \n
        'T68' is Trident
    shorten : Boolean
        if True returns a shortened version of trials containing only relevant data.  Default is False.
    Returns
    ----------
    trials : dataFrame
        A datafram of trials: the csv table loaded into a pandas dataframe for analysis
    dates : numpyArray
        An array of each unique session date found in the original dataset.  This is used for further sorting.
    '''
    if mCode == []:
        mCode = input('\nEnter 3 character monkey code (ex: T73): ')
        if len(mCode) != 3:
            raise Exception('error: monkey code requires a 3 character string.')
            pass

    fileName = mCode + '_allTrials.csv'

    trials = pd.read_csv(fileName)  #reading the dataset into the variable space
    trials.fillna(
        value=np.nan, inplace=True
    )  #replaces the none values to nans, so I can work with them
    trials.rename(columns={'date': 'sessionDate'}, inplace=True)

    trials['sessionDate'] = pd.to_datetime(
        trials['sessionDate'], format='%Y/%m/%d')
    trials['time'] = pd.to_datetime(
        trials['time'], format='%H:%M:%S')  #format time as datetime
    trials['sessionDate'] = trials['sessionDate'].dt.date
    trials['time'] = trials[
        'time'].dt.time  #set the date and time columns as datetime specific

    maskA = trials['GA_m1'] < 0
    maskB = trials['GB_m1'] < 0
    trials.loc[maskA, ['GA_m1', 'GA_p1']] = np.nan
    trials.loc[maskA, ['GA_Outcomes']] = 0
    trials.loc[maskB, ['GB_m1', 'GB_p1']] = np.nan
    trials.loc[maskB, ['GB_Outcomes']] = 0

    trials.at[trials.loc[trials.gambleChosen == '\x00'].index, ['gambleChosen'
                                                               ]] = np.nan
    trials.at[trials.loc[trials.errorType.isnull()].loc[
        trials.gambleChosen.notnull()].index, ['error']] = 0
    trials.at[trials.loc[trials.errorType.isnull()].loc[
        trials.gambleChosen.isnull()].index, ['errorType']] = 'trial error'

    trials['gambleA'] = trials[[
        'GA_m1', 'GA_p1', 'GA_m2', 'GA_p2', 'GA_m3', 'GA_p3'
    ]].values.tolist()
    trials['gambleA'] = trials.gambleA.apply(
        lambda l: [x for x in l if str(x) not in ['', 'nan', 'NaN']])
    trials['gambleB'] = trials[[
        'GB_m1', 'GB_p1', 'GB_m2', 'GB_p2', 'GB_m3', 'GB_p3'
    ]].values.tolist()
    trials['gambleB'] = trials.gambleB.apply(
        lambda l: [x for x in l if str(x) not in ['', 'nan', 'NaN']])
    trials['GA_ev'] = trials['gambleA'].apply(
        lambda x: sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]))
    trials['GB_ev'] = trials['gambleB'].apply(
        lambda x: sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]))

    trials['ml_outcomes_1'] = [
        x if x >= 0 else 0 for x in trials['ml_outcomes_1']
    ]
    trials['ml_outcomes_2'] = [
        x if x >= 0 else 0 for x in trials['ml_outcomes_2']
    ]

    ml_received = []
    sequenceFilename = []
    with tqdm(total=len(trials), desc='Fetching Trials') as pbar:
        for index, trial in (trials.iterrows()):
            if not pd.isnull(trial.gambleChosen):
                ml_received.append([trial.ml_outcomes_1,
                                    trial.ml_outcomes_2][(ab2int(
                                        trial.gambleChosen))[0]])
            else:
                ml_received.append(None)

            if trial.trialSequenceMode != 9020:
                sequenceFilename.extend([np.nan])
            else:
                sequenceFilename.extend([trial.sequenceFilename])
            pbar.update(1)

    trials['ml_received'] = ml_received
    trials['sequenceFilename'] = sequenceFilename

    ml_drank = []
    for date in trials.sessionDate.unique():
        ml_drank.extend(
            flatten([[0],
                     np.nancumsum(trials.loc[trials.sessionDate == date]
                                  .ml_received.tolist()).tolist()[0:-1]]))
    trials["ml_drank"] = ml_drank
    trials['monkey'] = mCode

    if shorten == True:
        trials = get_shortTrials(trials)
    if not noInfo:
        return trials.sort_index(), get_sessionInfo(trials)
    else:
        return trials.sort_index()


#%%
def fetch_Trials(mCode, dates='all', shorten=True, noInfo=False):
    '''
    fetch_trials gets a copy of the data csv table from the monkey's database

    Parameters
    ----------
    mCode : string
        'T73' is Tigger \n
        'U74' is Ugo \n
        'T68' is Trident
    dates : string or list of strings
        'all' gets all dates, one string gets the specific date, and a list gets between those dates.\n
        Needs to be in yyyy-mm-dd format.
    Returns
    ----------
    trials : dataFrame
        A dataframe of trials: the csv table loaded into a pandas dataframe for analysis
    info : dataFrame
        A dataFrame containing ml drank, sides selected, dates, errors, and trials indexes for individual sessions.
    '''
    import sqlite3
    from pathlib import Path
    home = str(Path.home())

    db = home + r'\Google Drive\Lab Data\database' + "\\" + mCode
    conn = sqlite3.connect(db)

    if type(dates) == str:
        if dates.lower() == 'all':  #gets all dates
            trials = pd.read_sql("SELECT * FROM Trials_Digital", conn)
        else:
            print('your date doesn\'t make sense')
            return
    elif type(dates) is list:
        if len(dates) == 1:
            trials = pd.read_sql(
                'SELECT * FROM Trials_Digital WHERE date IS  "' + dates[0] +
                '"', conn)
        elif len(dates) == 2:
            trials = pd.read_sql(
                'SELECT * FROM Trials_Digital WHERE date BETWEEN "' + dates[0] +
                '" AND "' + dates[1] + '"', conn)
        else:
            print('your date doesn\'t make sense')
            return
    conn.close()

    trials.drop_duplicates(
        subset=['date', 'time'], keep='first',
        inplace=True)  #removes trials that were recorded by error
    trials.fillna(
        value=np.nan, inplace=True
    )  #replaces the none values to nans, so I can work with them
    trials.rename(columns={'date': 'sessionDate'}, inplace=True)

    trials['sessionDate'] = pd.to_datetime(
        trials['sessionDate'], format='%Y/%m/%d')
    trials['time'] = pd.to_datetime(
        trials['time'], format='%H:%M:%S')  #format time as datetime
    trials['sessionDate'] = trials['sessionDate'].dt.date
    trials['time'] = trials[
        'time'].dt.time  #set the date and time columns as datetime specific

    maskA = trials['GA_m1'] < 0
    maskB = trials['GB_m1'] < 0
    trials.loc[maskA, ['GA_m1', 'GA_p1']] = np.nan
    trials.loc[maskA, ['GA_Outcomes']] = 0
    trials.loc[maskB, ['GB_m1', 'GB_p1']] = np.nan
    trials.loc[maskB, ['GB_Outcomes']] = 0

    trials.at[trials.loc[trials.gambleChosen == '\x00'].index, ['gambleChosen'
                                                               ]] = np.nan
    trials.at[trials.loc[trials.errorType.isnull()].loc[
        trials.gambleChosen.notnull()].index, ['error']] = 0
    trials.at[trials.loc[trials.errorType.isnull()].loc[
        trials.gambleChosen.isnull()].index, ['errorType']] = 'trial error'

    trials['gambleA'] = trials[[
        'GA_m1', 'GA_p1', 'GA_m2', 'GA_p2', 'GA_m3', 'GA_p3'
    ]].values.tolist()
    trials['gambleA'] = trials.gambleA.apply(
        lambda l: [x for x in l if str(x) not in ['', 'nan', 'NaN']])
    trials['gambleB'] = trials[[
        'GB_m1', 'GB_p1', 'GB_m2', 'GB_p2', 'GB_m3', 'GB_p3'
    ]].values.tolist()
    trials['gambleB'] = trials.gambleB.apply(
        lambda l: [x for x in l if str(x) not in ['', 'nan', 'NaN']])
    trials['GA_ev'] = trials['gambleA'].apply(
        lambda x: sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]))
    trials['GB_ev'] = trials['gambleB'].apply(
        lambda x: sum([x[i] * x[i + 1] for i in range(0, len(x), 2)]))

    trials['ml_outcomes_1'] = [
        x if x >= 0 else 0 for x in trials['ml_outcomes_1']
    ]
    trials['ml_outcomes_2'] = [
        x if x >= 0 else 0 for x in trials['ml_outcomes_2']
    ]

    ml_received = []
    sequenceFilename = []
    with tqdm(total=len(trials), desc='Fetching Trials') as pbar:
        for index, trial in (trials.iterrows()):
            if not pd.isnull(trial.gambleChosen):
                ml_received.append([trial.ml_outcomes_1,
                                    trial.ml_outcomes_2][(ab2int(
                                        trial.gambleChosen))[0]])
            else:
                ml_received.append(None)

            if trial.trialSequenceMode != 9020:
                sequenceFilename.extend([np.nan])
            else:
                sequenceFilename.extend([trial.sequenceFilename])
            pbar.update(1)

    trials['ml_received'] = ml_received
    trials['sequenceFilename'] = sequenceFilename

    ml_drank = []
    for date in trials.sessionDate.unique():
        ml_drank.extend(
            flatten([[0],
                     np.nancumsum(trials.loc[trials.sessionDate == date]
                                  .ml_received.tolist()).tolist()[0:-1]]))
    trials["ml_drank"] = ml_drank
    trials['monkey'] = mCode
    trials.rename(columns={'date': 'sessionDate'}, inplace=True)

    if shorten == True:
        trials = get_shortTrials(trials)
    if not noInfo:
        return trials.sort_index(), get_sessionInfo(trials)
    else:
        return trials.sort_index()


#%%
def get_shortTrials(trials):
    '''
    From the full trials dataFrame, get only the data required for behavioural analysis.

    Parameters
    ----------
    trials : dataFrame
        DataFrame of trials.
    Returns
    ----------
    shortTrials : dataFrame
        Dataframe with minimal data.  Reduces size of initial Trials file.

    '''
    shortTrials = trials[[
        'monkey', 'sessionDate', 'time', 'trialNo', 'blockNo', 'gambleA',
        'gambleB'
    ]]  #select initial shortTrials variables
    shortTrials = shortTrials.assign(
        outcomesCount=trials[['GA_Outcomes', 'GB_Outcomes']].values.tolist())

    if 'filteredRT' in trials.columns:
        shortTrials = pd.concat(
            [
                shortTrials, trials[[
                    'GA_ev', 'GB_ev', 'gambleChosen', 'trialSequenceMode',
                    'sequenceFilename', 'filteredRT', 'choiceTime',
                    'j_firstStimulus', 'error', 'errorType', 'ml_received',
                    'ml_drank'
                ]]
            ],
            axis=1)
    else:
        shortTrials = pd.concat(
            [
                shortTrials, trials[[
                    'GA_ev', 'GB_ev', 'gambleChosen', 'trialSequenceMode',
                    'sequenceFilename', 'choiceTime', 'j_firstStimulus',
                    'error', 'errorType', 'ml_received', 'ml_drank'
                ]]
            ],
            axis=1)

    shortTrials = pd.concat(
        [
            shortTrials, trials[[
                'j_leftGain', 'j_rightGain', 'j_xOffset', 'j_yOffset',
                'stimulusOn', 'j_onStimulus'
            ]]
        ],
        axis=1)
    print('Shortening...')

    return shortTrials.sort_index()


#%%
def get_sessionInfo(trials):
    '''
    Get information specific to each testing session.  I.e. Mls drank, sides chosen, errors, etc...
    Parameters
    ----------
    Trials : DataFrame
        Can be short or normal trials.
    Dates : list or array of datetimes
        Dates to gather information from (usually passed as the dates within the Trials DataFrame)
    Returns
    ----------
    info : dataFrame
        A dataFrame containing ml drank, sides selected, dates, errors, and trials indexes for individual sessions.
    '''
    from collections import Counter  #for the counter
    cols = [
        'date', 'ml_total', 'noOfChoices', 'leftChoice', 'rightChoice',
        'noChoice', 'iTrials', 'errors', 'SequenceBlocks', 'trialsPerBlock',
        'SequenceModes', 'SequenceTrials', 'options'
    ]
    #    iTrials = [];
    #    info.date =  Dates
    dfs = []
    Dates = trials.sessionDate.unique()
    for date in tqdm(
            Dates,
            desc='Gathering Session Information'):  #this is not fuckng working
        Count = Counter(
            trials.loc[trials.sessionDate == date].gambleChosen.tolist())
        df = trials.loc[trials.sessionDate == date]
        try:
            dfs.append(
                pd.DataFrame({
                    'date':
                    date,
                    'ml_total': [round(np.nansum(df.ml_received.tolist()), 2)],
                    'noOfChoices': [Count['A'] + Count['B']],
                    'leftChoice':
                    [round((Count['A'] / (Count['A'] + Count['B'])) * 100, 2)],
                    'rightChoice':
                    [round((Count['B'] / (Count['A'] + Count['B'])) * 100, 2)],
                    'noChoice': [Count[np.nan]],
                    'iTrials': [df.index.tolist()],
                    'errors': [dict(Counter(df.errorType.tolist()))],
                    'SequenceModes':
                    [dict(Counter(df.trialSequenceMode.tolist()))],
                    'SequenceTrials': [
                        dict(Counter(df.sequenceFilename.tolist()))
                    ],
                    'options':
                    [unique_listOfLists(df.gambleA.values + df.gambleB.values)],
                    'SequenceBlocks': [{
                        index: df.loc[df.blockNo == index]
                        .sequenceFilename.unique().tolist()
                        for index in df.blockNo.unique()
                    }],
                    'trialsPerBlock': [{
                        index: len(df.loc[df.blockNo == index])
                        for index in df.blockNo.unique()
                    }]
                }))
        except:
            print('for', date, 'there was no successful trials')
            pass
    info = pd.concat(dfs, ignore_index=True)
    print('\nFound', str(len(flatten(info.iTrials.tolist()))),
          'trials matching this/those dates')
    return info[cols]


#%%
def get_delError(trials, errorType='all'):
    '''
    From a trials dataFrame, remove specific or all error trials.

    Parameters
    ----------
    trials : dataFrame
        DataFrame of trials.
    errorType : string or list of strings
        'noTouch' \n
        'NoEffort'\n
        'noChoice error'\n
        'noCenter'\n
        'earlyMove'\n
        'wrongSideMove'\n
        'ControlHold error'\n
        'hold error'
        'trial error'

    Returns
    ------------
    filteredTrials : dataFrame
        Dataframe of trials with specified error trials removed.

    '''
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    errorStatus = []
    trials.at[trials.loc[trials.gambleChosen.apply(
        lambda x: x != 'A' and x != 'B')].index, 'error'] = 1
    trials.at[trials.loc[trials.gambleChosen.apply(
        lambda x: x != 'A' and x != 'B')].index, 'errorType'] = 'noChoice'
    #
    #    trials.error[trials.gambleChosen.apply(lambda x: x!='A' and x!='B')] = 1 #make sure that nan trials lead to error
    #    trials.errorType[trials.gambleChosen.apply(lambda x: x!='A' and x!='B')] = 'noChoice'

    if errorType == 'all':
        partialIndex = (
            trials.error == 1)  #indexes trials that have an error via boolean
        Index = [i for i, x in enumerate(partialIndex) if x]  #index in numbers
    else:
        if type(errorType) == str:
            errorType = list([errorType])
        Index = []
        for error in errorType:
            partialIndex = (trials.errorType == error)  #index in boolean
            Index.extend(
                [i for i, x in enumerate(partialIndex) if x])  #index in numbers
        Index.sort()

    print('\nRemoving', str(len(Index)), 'trials from the original',
          str(len(trials)))
    delTrials = trials.loc[Index].copy()
    trials.drop(trials.index[Index], inplace=True)  #drops the index rows

    print('\nSelecting the following errors')
    for ePrint in delTrials.errorType.unique():
        if type(ePrint) is float:
            print(
                '\'', str(ePrint), '\'', ' : ',
                str(sum(delTrials.errorType.apply(lambda x: type(x) is float))),
                ' trials removed')
        else:
            print('\'', str(ePrint), '\'', ' : ',
                  str(sum(delTrials.errorType == ePrint)), ' trials removed')

    trials = get_delUnfeasable(trials)

    #also provide a dataframe of trials that were delete - so I can look for patterns in the deleted data I might have missed
    return trials.sort_index(), delTrials


#%%
def add_chosenEV(trials):
    trials.loc[:,'chosenEV'] = [
        aEV
        if not ord(chosen.lower()) - 97 else bEV for chosen, aEV, bEV in zip(
            trials['gambleChosen'], trials['GA_ev'], trials['GB_ev'])
    ]
    return trials


#%%
def fetch_rawJoystick(trials, ignoreEye=True):
    '''
    '''
    mCode = trials.monkey.unique().item()

    import sqlite3
    import pandas as pd
    from pathlib import Path
    home = str(Path.home())

    db = home + r'\Google Drive\Lab Data\database' + "\\" + mCode
    conn = sqlite3.connect(db)

    dates = [dd.strftime("%Y-%m-%d") for dd in trials.sessionDate.unique()]
    analogTrials = []
    for dd, day in tqdm(
            zip(dates, trials.sessionDate.unique()),
            total=len(dates),
            desc='add analog signal'):
        df = trials.loc[trials.sessionDate == day]
        aTrials = pd.read_sql(
            'SELECT * FROM Trials_Analog WHERE date IS "' + dd + '"', conn)

        eyeX = aTrials['Joystick_X'].copy()
        eyeY = aTrials['Joystick_Y'].copy()
        aTrials[['Joystick_X', 'Joystick_Y']] = aTrials[['eye_X', 'eye_Y']]
        aTrials['eye_X'] = eyeX
        aTrials['eye_Y'] = eyeY

        if ignoreEye:
            aTrials.drop(columns=['eye_X', 'eye_Y'], inplace=True)

        aTrials.rename(columns={'date': 'sessionDate'}, inplace=True)
        aTrials['sessionDate'] = pd.to_datetime(
            aTrials['sessionDate'], format='%Y/%m/%d')
        aTrials['time'] = pd.to_datetime(
            aTrials['time'], format='%H:%M:%S')  #format time as datetime
        aTrials['sessionDate'] = aTrials['sessionDate'].dt.date
        aTrials['time'] = aTrials[
            'time'].dt.time  #set the date and time columns as datetime specific

        aTrials.drop(columns=['trialNo', 'blockNo'], inplace=True)

        #        aTrials.drop_duplicates(subset=['sessionDate', 'time'], keep = 'last', inplace=True) #removes trials that were recorded by error

        #drop the proper duplicates... the ones with errors
        dups = aTrials.loc[aTrials.duplicated(
            subset=['sessionDate', 'time'], keep=False)].analogTime
        dups2drop = [
            i + 1 if len(x) > len(dups[i + 1]) else i
            for i, x in zip(dups[::2].index, dups[::2].values)
        ]

        aTrials.drop(dups2drop, inplace=True)

        analogTrials.append(
            df.merge(aTrials, how='inner', on=['sessionDate', 'time']))

    conn.close()
    analogTrials = pd.concat(analogTrials, ignore_index=True)

    # ---------------------------------
    analogTrials = extract_analogSignal(analogTrials, ignoreEye)
    # ---------------------------------
    return analogTrials


#%%
def extract_analogSignal(analogTrials, ignoreEye=True):
    import json

    print('Decoding analog signal from JSON format')
    #    analogTrials['analogTime'] = analogTrials['analogTime'].apply(lambda x: json.loads(x.decode('utf8')))
    print('Decoding analogTime...')
    analogTrials['analogTime'] = analogTrials['analogTime'].apply(json.loads)
    #    analogTrials['Joystick_X'] = analogTrials['Joystick_X'].apply(lambda x: json.loads(x.decode('utf8')))
    print('Decoding Joystick_X...')
    analogTrials['Joystick_X'] = analogTrials['Joystick_X'].apply(json.loads)
    #    analogTrials['Joystick_Y'] = analogTrials['Joystick_Y'].apply(lambda x: json.loads(x.decode('utf8')))
    print('Decoding Joystick_Y...')
    analogTrials['Joystick_Y'] = analogTrials['Joystick_Y'].apply(json.loads)

    if not ignoreEye:
        print('Decoding eye_X...')
        analogTrials['eye_X'] = analogTrials['eye_X'].apply(json.loads)
        #            analogTrials['eye_X'] = analogTrials['eye_X'].apply(lambda x: json.loads(x.decode('utf8')))
        print('Decoding eye_Y...')
        analogTrials['eye_Y'] = analogTrials['eye_Y'].apply(json.loads)
#            analogTrials['eye_Y'] = analogTrials['eye_Y'].apply(lambda x: json.loads(x.decode('utf8')))

    return analogTrials


#%%
def get_responseTimes(aTrials, removeAnalog=True, debugPlot=False):
    '''
    '''
    from scipy import signal
    import matplotlib.pyplot as plt

    nTr = len(aTrials)
    RTj = [np.nan] * nTr

    # low-pass filter parameters (for joystick position)
    Nc = 200
    CFc = 15
    # threshold parameters
    dt0 = 0.1
    nSD = 3
    dtover = 0.1

    FSb = 500  #sampling freqency for behaviour
    jdt = 1 / FSb  #sampling frequency per seconds (500)
    Fs = 1 / jdt  #sampling frequency in Hz

    np.warnings.filterwarnings('ignore')

    #    import matlab.engine
    #    eng = matlab.engine.start_matlab()

    if debugPlot:
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 4))

    for tr, trial in tqdm(
            aTrials.iterrows(), total=len(aTrials), desc='Calculating RTs'):
        # joystick
        jx = np.array(trial['Joystick_X'])  #.JXYEXY(1,:) #get joystick input
        if len(jx) < 3 * Nc:
            continue
        tt = np.array(trial['analogTime'])

        LowPass_window = signal.firwin(Nc, CFc / (Fs / 2), window='hamming')
        #        LowPass_window = eng.fir1(matlab.double([Nc]), matlab.double([CFc/(Fs/2)]))
        #        LowPass_window = np.array(LowPass_window).tolist()[0]

        jxsm = signal.filtfilt(LowPass_window, 1, jx, axis=0)
        #        np.warnings.filterwarnings('default')
        jvxsm1 = np.gradient(jxsm, jdt)
        #what does this do?
        #        jvxsm1 = np.gradient(jxsm) #this will be the appropriate one (ish)
        # 5-point differentiation - still need to do this

        jgx = [-1000, 1000]
        jgx = jgx[0]  # this is almost always -1000 (always, in this case)
        cx = jxsm * jgx
        jgL = trial['j_leftGain']
        jgR = trial['j_rightGain']

        cx[jxsm < 0] = cx[jxsm < 0] * jgL
        cx[jxsm > 0] = cx[jxsm > 0] * jgR

        j_off = [trial['j_xOffset'],
                 trial['j_yOffset']]  #this is what the next one is
        cx = cx + j_off[0]
        cvxsm1 = np.gradient(cx, jdt)

        #timings for stimulus on and joystick entering stimulus area
        t1 = trial['stimulusOn']  # eT(eC==code_stimOnFlip);
        t4 = trial['j_onStimulus']  # eT(eC==code_inStim);

        # establish threshold for which movement is considered true
        ii = (tt > t1) & (tt < (t1 + dt0))
        v0 = jvxsm1[
            ii]  #only look at motion once stimulus has appeared to create the threshold
        m0 = np.mean(v0)
        sd0 = np.std(v0)
        thr1 = m0 + nSD * sd0
        thr2 = m0 - nSD * sd0

        if debugPlot:
            #            fig, ax = plt.subplots(1,1, squeeze=False, figsize=(10, 4))
            jvx = np.gradient(jx)
            ax[0, 0].plot(tt, 10 * jx, color='y', alpha=0.5)
            ax[0, 0].plot(tt, 10 * jxsm, linewidth=1, color='orange', alpha=0.5)
            #            ax[0,0].plot(tt,jvx, alpha = 0.2)
            ax[0, 0].plot(tt, jvxsm1, color='r', alpha=0.5)
            #            ax[0,0].plot(tt,0.001*cvxsm1)

            ax[0, 0].axvline(t1, color='k', alpha=0.2, linewidth=2)
            ax[0, 0].axvline(t4, color='k', alpha=0.2, linewidth=2)

            ii = np.ravel(np.where((tt > t1) & (tt < (t1 + dt0))))
            ax[0, 0].axhline(thr1, alpha=0.2, color='b')
            ax[0, 0].axhline(thr2, alpha=0.2, color='b')

        #find where the signal crosses the threshold
        dover = dtover / jdt
        ii = (tt > t1) & (tt < (t4 + 1))
        v = jvxsm1[ii]
        iiv = (v > thr1) | (v < thr2)
        iivc = np.cumsum(iiv)
        iivcd = iivc[int(dover):] - iivc[:int(
            -dover)]  #check to make sure this selects the correct indexing
        #what does this do?

        try:
            iover = np.where(iivcd == dover)[0][0]
        except:
            iover = np.nan
        rtj = iover * jdt

        if not np.isnan(rtj):
            RTj[tr] = rtj
            #this is the reaction time

    aTrials['filteredRT'] = RTj

    Compare = np.vstack((~np.isnan(aTrials.filteredRT.values),
                         ~np.isnan(aTrials.choiceTime.values)))
    iCompare = [i for i, x in enumerate(Compare.T) if x[0] != x[1]]

    for i in iCompare:
        aTrials.filteredRT.iloc[i] = aTrials.iloc[i].choiceTime

    if removeAnalog:
        aTrials.drop(
            columns=['analogTime', 'Joystick_X', 'Joystick_Y'], inplace=True)
    return aTrials


#189 of these are under 0


#%%
def delete_sqlAnalog(mCode):
    import sqlite3
    import os

    from pathlib import Path
    home = str(Path.home())

    db = home + r'\Google Drive\Lab Data\database' + "\\" + mCode
    conn = sqlite3.connect(db)

    cursor = conn.cursor()
    dropTableStatement = "DROP TABLE Trials_Analog"
    cursor.execute(dropTableStatement)

    conn.close()


#%%
def syncAnalog_toSQL(mCode):
    '''
    '''

    import sqlite3
    import os
    os.chdir(
        "C:\\Users\\phbuj\\University Of Cambridge\\OneDrive - University Of Cambridge\\Lab Computer\\Lab_data_B"
    )
    cwd = os.getcwd()
    fileIDs = []

    for file in os.listdir(cwd):
        if file.endswith(".mat") and file.startswith(mCode):
            fileIDs.append(file)

    from pathlib import Path
    home = str(Path.home())

    db = home + r'\Google Drive\Lab Data\database' + "\\" + mCode
    conn = sqlite3.connect(db)

    for fileID in fileIDs:
        import_analog(fileID, conn)
    conn.close()


#%%
def import_analog(fileID, conn):
    '''
    '''
    import pandas as pd
    import numpy as np
    import datetime
    from scipy.io import loadmat
    import json

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

#        print(str([data['hh'].data.Trials[n].analog_times.tolist()]))

        dfs.append(
            pd.DataFrame({
                'date': [year + '-' + month + '-' + day],
                'time':
                time,
                'trialNo':
                n,
                'blockNo':
                z,
                'analogTime': [data['hh'].data.Trials[n].analog_times.tolist()],
                'Joystick_X':
                [data['hh'].data.Trials[n].analog_data[:, 0].tolist()],
                'Joystick_Y':
                [data['hh'].data.Trials[n].analog_data[:, 1].tolist()],
                'eye_X': [data['hh'].data.Trials[n].analog_data[:, 2].tolist()],
                'eye_Y': [data['hh'].data.Trials[n].analog_data[:, 3].tolist()]
            }))

    print(fileID)
    if dfs == []:
        return
    analogDF = pd.concat(dfs, ignore_index=True)

    analogDF.analogTime = analogDF.analogTime.apply(
        lambda xx: json.dumps(xx).encode('utf8'))
    analogDF.eye_X = analogDF.eye_X.apply(
        lambda xx: json.dumps(xx).encode('utf8'))
    analogDF.eye_Y = analogDF.eye_Y.apply(
        lambda xx: json.dumps(xx).encode('utf8'))
    analogDF.Joystick_X = analogDF.Joystick_X.apply(
        lambda xx: json.dumps(xx).encode('utf8'))
    analogDF.Joystick_Y = analogDF.Joystick_Y.apply(
        lambda xx: json.dumps(xx).encode('utf8'))

    analogDF.to_sql('Trials_Analog', conn, if_exists='append', index=False)


#%%
def get_delUnfeasable(trials):
    total = len(trials)
    primaryOptions = unique_listOfLists(trials.gambleA.values.tolist())
    wrong = [
        option for option in primaryOptions
        if np.sum(np.array(option[1::2])) != 1.00
    ]
    secondaryOptions = unique_listOfLists(trials.gambleB.values.tolist())
    wrong.extend([
        option for option in secondaryOptions
        if np.sum(np.array(option[1::2])) != 1.00
    ])

    for option in wrong:
        if option != []:
            trials.drop(
                trials.loc[trials.gambleA.apply(lambda x: x == option)].index,
                inplace=True)
            trials.drop(
                trials.loc[trials.gambleB.apply(lambda x: x == option)].index,
                inplace=True)

    print('\nRemoving an extra', str(len(flatten(wrong))),
          'trials from the original', str(total),
          '\nThey were unfeasible: probabilities did not add to 1.')
    return trials
