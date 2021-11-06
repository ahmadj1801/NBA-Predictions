import numpy as np
import pandas as pd
import sklearn
import os

from sklearn.neural_network import MLPRegressor

START = 1997
END = 2005


def read_data(path):
    """
    Returns a DataFrame with the read in data
    :param path: Path to data file
    :return: DataFrame
    """
    data = pd.read_csv(path)
    return data


def pre_process_data(data, start=1997, end=2005):
    """
    Store the data by year, as well as convert the stats to per game stats.
    Example: {2002: DataFrame, 2003: DataFrame}
    :param data: The raw dataframe
    :param start: From when to start considering
    :param end: The year to stop at. Stops at end -1
    :return: Dict of DataFrames
    """
    data_dict = dict()
    for i in range(start, end):
        season_data = data[data['year'] == i]
        total_games = season_data.iloc[0]['won'] + season_data.iloc[0]['lost']
        columns = season_data.columns
        for j in range(3, len(columns)):
            col = columns[j]
            season_data[col] = season_data[col]/total_games
        season_data = season_data.drop(columns=['team', 'year', 'leag', 'pace'], axis=1)
        data_dict[i] = season_data
    return data_dict


def offensive_defensive_data(data_dict, cols):
    """
    Retrieve the data in a combined DataFrame containing only the data needed by the cols parameter
    :param data_dict: Dictionary of DataFrames
    :param cols: Column headers
    :return: A DataFrame
    """
    df = pd.DataFrame(columns=cols)
    for key in data_dict:
        d = data_dict[key]
        d = d[cols].copy()
        df = pd.concat([df, d], axis=0)
    return df


def main():
    """
    Main Code
    :return: No Return
    """

    # Read data
    data = read_data('../Data/team_season.txt')

    # Get 2004 Data only
    data_2004 = data[data['year'] == 2004]

    # Get the Teams participating in the 2004 season
    teams = data_2004.team.tolist()

    # Pre-processing
    data_dict = pre_process_data(data, start=START, end=END)

    # Attacking columns we are interested in
    attack_cols = ['o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb', 'o_dreb',
                   'o_reb', 'o_asts', 'o_pf', 'o_stl', 'o_to', 'o_blk', 'o_3pm', 'o_3pa', 'o_pts']

    # DataFrame with attacking data only
    attacking_df = offensive_defensive_data(data_dict, attack_cols)

    # Attacking data for the teams in the 2004 season
    attack_2004 = attacking_df.tail(len(teams))

    # Defensive columns we are interested in
    def_cols = ['d_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_oreb', 'd_dreb', 'd_reb',
                'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_3pm', 'd_3pa',
                'd_pts']

    # DataFrame with defensive data only
    def_df = offensive_defensive_data(data_dict, def_cols)

    # Defensive data for the teams in the 2004 season
    defense_2004 = def_df.tail(len(teams))

    # Columns needed for the win loss predictor
    win_loss_cols = ['o_pts', 'd_pts', 'won']

    # DataFrame containing offensive and defense points that will be used to compute a confidence
    win_df = offensive_defensive_data(data_dict, win_loss_cols)

    # Attacking Network
    attack_mlp = MLPRegressor().fit(attacking_df.drop(columns=['o_pts']), attacking_df['o_pts'])

    # Defensive Network
    defense_mlp = MLPRegressor().fit(def_df.drop(columns=['d_pts']), def_df['d_pts'])

    """
    TODO: Setup AdaBoost Regressor for confidence prediction
    TODO: Facilitate the loop to ask for 2 input teams
    TODO: Define a mechanism to smoothly capture the necessary features for prediction
    """
    pass


if __name__ == '__main__':
    main()
