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

    my_algo = foxes.algorithms.Downwind(
        mbook=my_mbook,
        farm=my_farm,
        states=my_states,
        rotor_model=rotor_model,
        wake_models=wake_models,
        verbosity=0,
    )
    return my_algo