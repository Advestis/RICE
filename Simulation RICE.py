import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import RICE


def make_y(x, noise, th_min=-0.4, th_max=0.4):
    y_vect = [-2 if x_val <= th_min else
              0 if x_val <= th_max else
              2 for x_val in x]
    y_vect += np.random.normal(0, noise, len(y_vect))
    return np.array(y_vect)


def make_condition(rg):
    rep = ''
    for i in range(len(rg['Features_Name'])):
        if i > 0:
            rep += ' & '
        rep += rg['Features_Name'][i]
        if rg['BMin'][i] == rg['BMax'][i]:
            rep += ' = '
            rep += str(rg['BMin'][i])
        else:
            rep += ' $\in$ ['
            rep += str(rg['BMin'][i])
            rep += ', '
            rep += str(rg['BMax'][i])
            rep += ']'
    return rep


nRows = 5000
nCols = 2
noise = 1.0
h = 0.05

np.random.seed(42)
X = np.random.uniform(low=-1, high=1,
                      size=(nRows, nCols))

cm = plt.cm.coolwarm  # plt.cm.binary


def test_min(bins, df, rg_id):
    vars_name = df.loc[rg_id, 'Features_Name']
    
    i = 0
    bmin = tuple()
    for v in vars_name:
        min_i = df.loc[rg_id, 'BMin'][i]
        if int(min_i) != 0:
            bmin += (bins[v][int(min_i) - 1],)
        else:
            bmin += (-1,)
    
    return bmin


def test_max(bins, df, rg_id, nb_bucket):
    vars_name = df.loc[rg_id, 'Features_Name']
    
    i = 0
    bmax = tuple()
    for v in vars_name:
        max_i = df.loc[rg_id, 'BMax'][i]
        if int(max_i) != nb_bucket - 1:
            bmax += (bins[v][int(max_i)],)
        else:
            bmax += (1,)
    
    return bmax


def Mrn(rg1, rs):
    act = copy.copy(rg1.get_param('activation'))
    for rg2 in rs:
        act = map(lambda val1, val2: 0 if val1 == 0 else val1 + val2, act, rg2.get_param('activation'))
        act = np.array(act)
    return max(act) - 1


x_vect = X[:, 0]
th_min = -0.4
th_max = 0.4

y = make_y(x_vect, noise, th_min, th_max)

test_size = 0.40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=42)

rice_lin = RICE.Learning()

rice_lin.fit(X_train, y_train)

x_min, x_max = X[:, 0].min(), X[:, 0].max() + h  # add left and right margins
y_min, y_max = X[:, 1].min(), X[:, 1].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

ZZ = rice_lin.predict(np.array(list(zip(xx.ravel(), yy.ravel()))), check_input=False)
ZZ = ZZ.reshape(xx.shape)
