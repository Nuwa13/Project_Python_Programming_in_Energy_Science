import pandas as pd
import numpy as np
import foxes
import foxes.variables as FV
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error,mean_squared_error
import warnings

scaler = StandardScaler()

def read_ts_csv(filename):
    df = pd.read_csv(filename, parse_dates=['Unnamed: 0'])
                # data
    df.rename(columns={ df.columns[0]: "Time" }, inplace = True)
                # rename the first column to be the 'time' column
    df.drop_duplicates(subset='Time', inplace=True)

                # check for -999 and replace with NaN if applicable
    if (df == -999).any().any():
        df.replace(-999, np.nan, inplace=True)
        
    return df

def compute_yield(algo):
    results = algo.calc_farm(calc_parameters={"chunk_size_states": 1000})
    # how to call inside script:
    # from customFunctions import compute_yield

    # turbine_stats, summary = compute_yield(results, algo)
    # print(turbine_stats)
    # print(summary)
    eval_ = foxes.output.FarmResultsEval(results)
    eval_.add_capacity(algo)
    eval_.add_capacity(algo, ambient=True)
    eval_.add_efficiency()

    # Compute per-turbine annual yields
    yld_net_da = eval_.calc_turbine_yield(algo, annual=True)
    yld_amb_da = eval_.calc_turbine_yield(algo, annual=True, ambient=True)

    yld_net = yld_net_da.values.squeeze()
    yld_amb = yld_amb_da.values.squeeze()
    eff_per = yld_net / yld_amb

    turbine_stats = pd.DataFrame({
        "Ambient Yield [GWh]": yld_amb,
        "Net Yield     [GWh]": yld_net,
        "Efficiency         ": eff_per
    }, index=algo.farm.turbine_names)

    # Farm-level summary
    farm_ambient = eval_.calc_mean_farm_power(ambient=True) / 1000  # MW
    farm_net     = eval_.calc_mean_farm_power()          / 1000     # MW
    farm_eff     = eval_.calc_farm_efficiency()                   # unitless
    annual_yld   = yld_net.sum()                                  # GWh

    summary = {
        "farm_ambient_power_MW": farm_ambient,
        "farm_net_power_MW": farm_net,
        "farm_efficiency": farm_eff,
        "annual_yield_GWh": annual_yld
    }

    return turbine_stats, summary



def setup_algo(wind_data, windfarm_name = 'my_farm',rotor_model = "centre", TI = 0.05, RHO = 1.225, wake_models = ["Jensen_linear_k004","IECTI2019k_linear_k004"],model_book = None,layout_data = None,turbine_key = None):
    my_farm = foxes.WindFarm(name=windfarm_name)

    if model_book is None:
        my_mbook = foxes.models.ModelBook()
    else:
        my_mbook = model_book

    if turbine_key is None:
        my_turbine_key = ["DTU10MW"]
    else:
        my_turbine_key = turbine_key

    if layout_data is None:
        layout_data = pd.read_csv('turbine-info/coordinates/area_of_interest/layout-N-10.1.geom.csv',
                              index_col='Unnamed: 0').sort_values(by='y', ascending=False).reset_index(inplace=False)
    foxes.input.farm_layout.add_from_csv(my_farm, layout_data, turbine_models=my_turbine_key)

    if 'WS' not in '\t'.join(wind_data.columns.values):
        warnings.warn('setup_windfarm failed: wind speed data should be named "WS" but there is no such column in the input data!')
    else:
        WS_col = next((s for s in wind_data.columns.values if 'WS' in s), None)
        
    if 'WD' not in '\t'.join(wind_data.columns.values):
        warnings.warn('setup_windfarm failed: wind speed data should be named "WD" but there is no such column in the input data!')
    else:
        WD_col = next((s for s in wind_data.columns.values if 'WD' in s), None)

    # if 'WS' not in wind_data.columns:
    #     print('setup_windfarm failed: wind speed data should be named "WS" but there is no such column in the input data!')
    # if 'WD' not in wind_data.columns:
    #     print(
    #         'setup_windfarm failed: wind speed data should be named "WD" but there is no such column in the input data!')
    my_states = foxes.input.states.Timeseries(
        data_source=wind_data,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: WS_col, FV.WD: WD_col},
        fixed_vars={FV.RHO: RHO, FV.TI: TI},
    )

    my_algo = foxes.algorithms.Downwind(
        mbook=my_mbook,
        farm=my_farm,
        states=my_states,
        rotor_model=rotor_model,
        wake_models=wake_models,
        verbosity=0,
    )
    return my_algo


def print_performance(y_predicted,y_true, title=''):
    print('Validation scores - ' + title + ':')
    print('-------------------------')
    print('R2-Score: ',round(r2_score(y_true, y_predicted),4))
    print('MSE: ',round(mean_squared_error(y_true, y_predicted), 4))
    print('RMSE: ',round(root_mean_squared_error(y_true, y_predicted), 4))
    print('MAE: ',round(mean_absolute_error(y_true, y_predicted), 4))
    print('MAPE: ',round(mean_absolute_percentage_error(y_true, y_predicted)*100, 4))
    


def correct_long_term_wind(wind_model, wind_measurement, classifier, param_grid):
    # choose a value for comparability
    RSEED = 42

    # define how many percent of data should be excluded from training to use it for testing
    test_size = 0.3
    
    # set variables 
    # model data = Features 
    X = pd.DataFrame(wind_model)

    # valid values = observations
    Y = pd.DataFrame(wind_measurement)

    # remove nan values
    # find index of nan values
    idx = Y[Y.isnull().any(axis=1)].index

    #find index of valid values
    meas_idx = Y[Y.notnull().all(axis=1)].index
    meas_idx = [wind_measurement.index.get_loc(value) for value in meas_idx]

    XtoPredict = X.loc[idx]

    # features
    x = X.iloc[meas_idx]

    # target
    y = Y.iloc[meas_idx]
    xcols = X.columns
    
    # split data into training and test data (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=RSEED)

    # scale data to a common basis (e.g. WS ranges from 0-30 m/s while wind direction from 0 to 360, scaling fits them into a common range)
    # scaling is done based on trainings data (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    X_train_scaled = scaler.fit_transform(X_train)

    # scaling procedure is transfered to test data
    X_test_scaled = scaler.transform(X_test)
    
    # how many times shall the cross validation be done
    n          = 5

    # according to which metrics shall the performance be tested
    scoring    = 'neg_mean_squared_error'

    # train the model
    # GridSearch finds the best parameter combination for the model for which the score is minimal. 
    # A cross validation is done to compare results from different subsets, 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    grid_model = GridSearchCV(classifier, param_grid=param_grid, cv=n, scoring=scoring,\
                                    verbose=0, n_jobs=-1).fit(X_train_scaled, y_train)
    
    best_model = grid_model.best_estimator_
    print('Best score:\n{:.2f}'.format(grid_model.best_score_))
    print("Best parameters:\n{}".format(grid_model.best_params_))

    # make predictions using the trainings to check model performance
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
    y_train_cv      = cross_val_predict(best_model, X_train_scaled, y_train, cv=n+2)

    # predict values using trained model and test data (features)
    y_test_predicted = best_model.predict(X_test_scaled)

    # make your final prediction for the entire (long-term) period
    X_scaledN = scaler.transform(X)
    
    # print_performance(y_train_cv,y_train, 'Cross Validation')
    print_performance(x.iloc[:, 0], y.iloc[:, 0], 'Without Correction')
    print_performance(y_test_predicted, y_test, 'Model and Test Data')

        
    return best_model.predict(X_scaledN), meas_idx

def wind_components(wind_vel, wind_dir):
    # check whether the direction is given in degree or radians:
    if np.nanmax(wind_dir) > 2 * np.pi:
        # convert degree into radian
        wind_dir = np.deg2rad(wind_dir)
        
    u = - np.abs(wind_vel) * np.sin(wind_dir)
    v = - np.abs(wind_vel) * np.cos(wind_dir)
    
    return u, v

def wind_speed_dir(u, v):
    
    wind_speed = np.sqrt(u**2 + v**2)
    wind_dir = (np.arctan2(u, v) * 180 / np.pi + 180) % 360
    
    return wind_speed, wind_dir