import pandas as pd
import numpy as np

from foxes.output import FarmResultsEval
import pandas as pd

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

def compute_yield(results, algo): 

    # how to call inside script: 
    # from customFunctions import compute_yield

    # turbine_stats, summary = compute_yield(results, algo)
    # print(turbine_stats)
    # print(summary)

    eval_ = FarmResultsEval(results)
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

