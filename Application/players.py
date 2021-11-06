import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import os
import argparse

START = 2000


def split_data(standard_season, team_standard_season):

    years = [i for i in range(1946, 2005)]
    num_players_per_year = standard_season['year'].value_counts()
    num_teams_per_year = team_standard_season['year'].value_counts()

    '''SPLIT INTO INDIVIDUAL YEARS'''
    original_season_dict = dict()
    season_dict = dict()
    total = 0
    for i in range(len(years)):
        num_players = num_players_per_year[years[i]]
        df = standard_season[total: total + num_players]
        original = df
        df = df.drop(['ilkid', 'year', 'firstname', 'lastname'], axis=1)
        # Add an attribute - Points per game
        df['ppg'] = df['pts'] / df['gp']
        # Add an attribute - Assists per game
        df['apg'] = df['asts'] / df['gp']
        # Add an attribute G/A per game
        df['cpg'] = (df['pts'] + df['asts']) / df['gp']
        # Add attribute - average field goals made
        df['afgm'] = df['fgm'] / df['fga']
        # Add attribute - average free throw made
        df['aftm'] = df['ftm'] / df['fta']
        # Add attribute - average three points made
        df['atpm'] = df['tpm'] / df['tpa']
        # Add attribute - Minutes per game
        df['mpg'] = df['minutes'] / df['gp']
        df = df.fillna(0)
        cols = df.columns
        df = MinMaxScaler().fit_transform(df[cols])
        df = pd.DataFrame(df, columns=cols)
        total += num_players
        season_dict[years[i]] = df
        original_season_dict[years[i]] = original
    return years, num_players_per_year, num_teams_per_year, original_season_dict, season_dict


def detect_best_players(year, original_season_dict, season_dict, all_stars):
    if year < 2004:
        all_stars_03 = all_stars[all_stars['year'] == year]

        all_stars_id = all_stars_03.ilkid.tolist()
        all_star_names = all_stars_03.firstname.tolist() + all_stars_03.lastname.tolist()

    original = original_season_dict[year]
    original = original.reset_index()
    data = season_dict[year]
    data = data.fillna(0)

    names = original.firstname.tolist()
    surnames = original.lastname.tolist()

    clf = IsolationForest(contamination=0.1)
    pred = clf.fit_predict(data)

    pca = PCA(n_components=2).fit(data)
    pca_2d = pca.transform(data)
    pca_x = []
    pca_y = []

    c1 = c2 = c3 = None
    for i in range(0, pca_2d.shape[0]):
        pca_x.append(pca_2d[i, 0])
        pca_y.append(pca_2d[i, 1])
        if pred[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='+')
        elif pred[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='o')
        elif pred[i] == -1:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')
    plt.title(str(year) + 'Before Modification')
    plt.savefig('Output/Year '+str(year)+'/'+str(year) + '.png')
    plt.clf()

    df = original
    df['x'] = pca_x
    df['y'] = pca_y

    if year < 2004:
        correct = 0
        total = 0
        for i in range(len(pred)):
            if pred[i] == -1:
                if names[i] in all_star_names:
                    correct += 1
                total += 1
            else:
                c2 = plt.scatter(pca_x[i], pca_y[i], c='r', marker='o')
                df = df.drop(i)
    else:
        for i in range(len(pred)):
            if not pred[i] == -1:
                c2 = plt.scatter(pca_x[i], pca_y[i], c='r', marker='o')
                df = df.drop(i)

    attributes = ['pts', 'asts', 'oreb', 'dreb', 'stl', 'blk', 'turnover']

    score = [0 for i in range(len(df))]
    ids = df.ilkid.tolist()
    for attribute in attributes:
        sorted_df = df.sort_values(attribute)
        value = 1
        for ind in df.index:
            id = sorted_df['ilkid'][ind]
            id_index = ids.index(id)
            score[id_index] = score[id_index] + value
            value += 1

    # Contains all our info that we need on our players
    df['scores'] = score
    df = df.sort_values('scores', ascending=False)
    df = df[df['gp'] < 83]

    ids = df.ilkid.tolist()

    c = 0
    is_all_star = []

    if year < 2004:
        all_stars_id = [(id.lower()).strip() for id in all_stars_id]
        ids = [id.lower() for id in ids]
        for id in ids[0:10]:
            if id in all_stars_id:
                c += 1
                is_all_star.append('Yes')
            else:
                is_all_star.append('No')

    pca_x = df.x.tolist()
    pca_y = df.y.tolist()
    for i in range(len(pca_x)):
        if i < 10:
            c1 = plt.scatter(pca_x[i], pca_y[i], c='b', marker='*')
        else:
            c2 = plt.scatter(pca_x[i], pca_y[i], c='r', marker='o')
    plt.title(str(year) + 'After Modification')
    plt.savefig('Output/Year ' + str(year) + '/' + str(year) + '_mod.png')
    plt.clf()

    df = df[0:10]
    if year < 2004:
        print('\n== Year {} == \n{} of the predicted 10 were part of the All Stars in the selected season'.format(year, c))
    else:
        print('\n== Year {} == \nNo All Star data for the selected season'.format(year, c))

    return df


def correlation_diagram(df):
    # Correlation Matrix Heat map
    f, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="Spectral", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('Correlation of Basketball Stats', fontsize=14)
    plt.savefig('Output/correlation.png')
    plt.show()


def line_graph(title, x_title, y_title, x, y):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.savefig('Output/' + title + '.png')
    plt.show()


def main():
    standard_season = pd.read_csv('../Data/player_regular_season.txt')
    standard_season = standard_season.groupby(['ilkid', 'year', 'firstname', 'lastname'], as_index=False).sum()
    standard_season = standard_season.sort_values('year')
    team_standard_season = pd.read_csv('../Data/team_season.txt')

    all_stars = pd.read_csv('../Data/player_allstar.txt')

    years, num_players_per_year, num_teams_per_year, original_season_dict, season_dict = split_data(standard_season,
                                                                                                    team_standard_season)

    num_players_per_year = [num_players_per_year[year] for year in years]
    num_teams_per_year = [num_teams_per_year[year] for year in years]

    correlation_diagram(original_season_dict[2004])
    # We can then drop those 6 columns - Highly correlated

    for key in original_season_dict:
        original_season_dict[key] = original_season_dict[key].drop(columns=['fgm', 'fga', 'ftm', 'fta', 'tpm', 'tpa'],
                                                                   axis=1)

    correlation_diagram(original_season_dict[2004])

    line_graph("Number of Players per Year", "Year", "Player Count", years, num_players_per_year)

    line_graph("Teams Registered per Year", "Year", "Team Count", years, num_teams_per_year)

    years = [i for i in range(START, 2005)]

    for year in years:
        does_exist = os.path.exists('Output/Year '+str(year))
        if not does_exist:
            os.mkdir('Output/Year '+str(year))
        df = detect_best_players(year, original_season_dict=original_season_dict,  season_dict=season_dict,
                                 all_stars=all_stars)
        path = 'Output/Year ' + str(year) + '/Top 10 Players of ' + str(year) + '.csv'
        df = df[['firstname', 'lastname', 'pts', 'asts', 'oreb', 'dreb', 'stl', 'blk', 'turnover']]
        df.to_csv(path)
        print('''{}'s Top 10 Players have been Stored in the directory -> {}'''.format(year, path))


if __name__ == '__main__':
    main()
