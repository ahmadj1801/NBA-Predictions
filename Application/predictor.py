import numpy as np
import pandas as pd
import sklearn
import os

from sklearn.neural_network import MLPRegressor

START = 1997
END = 2005


def read_data(path):
    data = pd.read_csv(path)
    return data


def pre_process_data(data, start=1997, end=2005):
    data_dict = dict()
    for i in range(start, end):
        season_data = data[data['year' == i]]


def main():
    pass


if __name__ == '__main__':
    main()
