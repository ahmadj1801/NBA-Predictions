import numpy as np
import pandas as pd
import sklearn
import argparse

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

START = 1980
END = 2005


def read_data(path):
    """
    Returns a DataFrame with the read in data
    :param path: Path to data file
    :return: DataFrame
    """
    data = pd.read_csv(path)
    return data


def pre_process_data(data, start=1980, end=2005):
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


def team_display(teams):
    display = 'Select an option below or enter -1 to exit:\n'
    for i in range(len(teams)):
        if i % 5 == 0:
            display = display + '\n{} - {}'.format(str(i), teams[i])
        else:
            display = display + '\t\t{} - {}'.format(str(i), teams[i])
    return display


def is_exit(a):
    if a == -1:
        exit()


def evaluate(results, team_dict, attack_mlp, defence_mlp, confidence_predictor, attack_2004, defence_2004, teams):
    visitor = results['Visitor/Neutral'].tolist()
    visitor_pts = results['PTS'].tolist()
    home = results['Home/Neutral'].tolist()
    home_pts = results['PTS.1'].tolist()
    winner = []
    for key in team_dict:
        for i in range(len(home)):
            key_name = team_dict[key]
            if key_name in home[i]:
                home[i] = key
            if key_name in visitor[i]:
                visitor[i] = key

    for i in range(len(home)):
        if home_pts[i] >= visitor_pts[i]:
            winner.append(home[i])
        else:
            winner.append(visitor[i])

    correct = 0
    for i in range(len(home)):
        home_team = home[i]
        away_team = visitor[i]
        team_a = teams.index(home_team)
        team_b = teams.index(away_team)
        team_a_win_confidence, team_b_win_confidence = predict_outcome(team_a, team_b, attack_mlp, defence_mlp, confidence_predictor, attack_2004, defence_2004, teams, team_dict)
        game_winner = ''
        print('Winner: {}'.format(winner[i]))
        if team_a_win_confidence[0] >= team_b_win_confidence[0]:
            print('Predicted Winner: {}'.format(teams[team_a]))
            game_winner = home_team
        else:
            print('Predicted Winner: {}'.format(teams[team_b]))
            game_winner = away_team
        if game_winner == winner[i]:
            correct += 1
    return correct/len(home)


def predict_outcome(team_a, team_b, attack_mlp, defense_mlp, confidence_predictor, attack_2004, defense_2004, teams, team_dict):
    team_a_attack = attack_2004.iloc[team_a]
    team_a_def = defense_2004.iloc[team_a]
    team_b_attack = attack_2004.iloc[team_b]
    team_b_def = defense_2004.iloc[team_b]
    # Points that A will score -> Whats B's defense will allow
    team_a_pts_scored = attack_mlp.predict([team_b_def])
    # Points that B will score -> What A's defense will allow
    team_b_pts_scored = attack_mlp.predict([team_a_def])
    # Points that A will concede -> According to B's attack on Defence
    team_a_pts_con = defense_mlp.predict([team_b_attack])
    # Points B will concede -> According to A's attack on defence
    team_b_pts_con = defense_mlp.predict([team_a_attack])
    team_a_match = [round(team_a_pts_scored[0]), round(team_a_pts_con[0])]
    team_b_match = [round(team_b_pts_scored[0]), round(team_b_pts_con[0])]

    team_a_win_confidence = confidence_predictor.predict([team_a_match])
    team_b_win_confidence = confidence_predictor.predict([team_b_match])
    print('''\n===Match Prediction===\n{} ({}) VS {} ({})'''.format(teams[team_a], team_dict[teams[team_a]],
                                                                  teams[team_b], team_dict[teams[team_b]]))
    print('''According to {}, the score will be {} {} : {} {}'''.format(teams[team_a], teams[team_a],
                                                                        round(team_a_pts_scored[0]),
                                                                        round(team_a_pts_con[0]), teams[team_b]))
    print('''According to {}, the score will be {} {} : {} {}'''.format(teams[team_b], teams[team_a],
                                                                        round(team_b_pts_con[0]),
                                                                        round(team_b_pts_scored[0]), teams[team_b]))
    print('''Given the scoreline, {} think they have a confidence of {} of winning the game'''.format(teams[team_a],
                                                                                                      team_a_win_confidence[0]))
    print('''Given the scoreline, {} think they have a confidence of {} of winning the game'''.format(teams[team_b],
                                                                                                      team_b_win_confidence[0]))
    return team_a_win_confidence, team_b_win_confidence


def main():
    """
    Main Code
    :return: No Return
    """

    parser = argparse.ArgumentParser(description='Outstanding Player Detection')
    parser.add_argument('--start', type=int, default=2000, help='The year to start collecting data from')
    parser.add_argument('-f')
    arguments = parser.parse_args()
    global START
    START = arguments.start

    # Read data
    data = read_data('../Data/team_season.txt')

    results = read_data('../Data/Winners2004.txt')

    team_data = read_data('../Data/teams.txt')

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
    attack_mlp = MLPRegressor(solver='lbfgs', max_iter=1000, early_stopping=True).fit(attacking_df.drop(columns=['o_pts']), attacking_df['o_pts'])

    # Defensive Network
    defense_mlp = MLPRegressor(solver='lbfgs', max_iter=1000, early_stopping=True).fit(def_df.drop(columns=['d_pts']), def_df['d_pts'])

    # Use ensemble techniques to create a regression confidence predictor
    confidence_predictor = AdaBoostRegressor(base_estimator=MLPRegressor(solver='lbfgs'), n_estimators=60,
                                             learning_rate=0.7, loss='square', random_state=43)

    # Fit the attacking and conceding points in relation to the winning ratio.
    confidence_predictor.fit(win_df.drop(columns=['won']), win_df['won'])

    attack_svr = SVR().fit(attacking_df.drop(columns=['o_pts']), attacking_df['o_pts'])

    # Defensive Network
    defense_svr = SVR().fit(def_df.drop(columns=['d_pts']), def_df['d_pts'])

    # Use ensemble techniques to create a regression confidence predictor
    confidence_predictor2 = AdaBoostRegressor(base_estimator=SVR(), n_estimators=50, learning_rate=0.7,
                                              loss='square', random_state=43)

    # Fit the attacking and conceding points in relation to the winning ratio.
    confidence_predictor2.fit(win_df.drop(columns=['won']), win_df['won'])

    team_dict = dict()
    for j in range(len(teams)):
        team = teams[j]
        names = team_data['name'].tolist()
        name = names[team_data['team'].tolist().index(team.upper())]
        team_dict[team] = name

    attack_2004 = attack_2004.drop(columns=['o_pts'], axis=1)
    defense_2004 = defense_2004.drop(columns=['d_pts'], axis=1)
    team_a = 0
    team_b = 0

    accuracy = evaluate(results, team_dict, attack_mlp, defense_mlp, confidence_predictor, attack_2004, defense_2004,
                        teams)
    accuracy_svr = evaluate(results, team_dict, attack_svr, defense_svr, confidence_predictor2, attack_2004,
                            defense_2004, teams)
    print('Accuracy of MLP =', accuracy)
    print('Accuracy of SVR =', accuracy_svr)

    while not team_a == -1 or not team_b == -1:
        print('\n', team_display(teams))
        team_a = int(input('Input the Home Team Number from the list above: '))
        is_exit(team_a)
        team_b = int(input('Input the Away Team Number from the list above: '))
        is_exit(team_b)
        team_a_win_confidence, team_b_win_confidence = predict_outcome(team_a, team_b, attack_mlp, defense_mlp, confidence_predictor, attack_2004, defense_2004, teams, team_dict)
        if team_a_win_confidence[0] >= team_b_win_confidence[0]:
            print('Winner: {}'.format(teams[team_a]))
        else:
            print('Winner: {}'.format(teams[team_b]))

    pass


if __name__ == '__main__':
    main()
