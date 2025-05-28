import pandas as pd
import numpy as np
import foxes
import foxes.variables as FV

def read_ts_csv(filename):
    df = pd.read_csv(filename, parse_dates=['Unnamed: 0'])
                # data
    df.rename(columns={ df.columns[0]: "Time" }, inplace = True)
                # rename the first column to be the 'time' column
    df.drop_duplicates(subset='Time', inplace=True)

                # check for -999 and replace with NaN if applicable
    if -999 in df:
        df[df==-999] = np.nan
        
    return df

def setup_windfarm(wind_data, windfarm_name = 'my_farm', TI = 0.05, RHO = 1.225, wake_models = ["Bastankhah2014_linear_k004"],model_book = None):
    my_farm = foxes.WindFarm(name=windfarm_name)
    if model_book is None:
        my_mbook = foxes.models.ModelBook()
    else:
        my_mbook = model_book

    if 'WS' not in wind_data.columns:
        print('setup_windfarm dailed: wind speed data should be named "WS" but there is no such column in the input data!')
    if 'WD' not in wind_data.columns:
        print(
            'setup_windfarm dailed: wind speed data should be named "WD" but there is no such column in the input data!')
    my_states = foxes.input.states.Timeseries(
        data_source=wind_data,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS", FV.WD: "WD"},
        fixed_vars={FV.RHO: RHO, FV.TI: TI},
    )

    algo = foxes.algorithms.Downwind(
        mbook=my_mbook,
        farm=my_farm,
        states=my_states,
        rotor_model="centre",
        wake_models=wake_models,
        verbosity=0,
    )
    return my_farm