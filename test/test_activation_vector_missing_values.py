from dotenv import load_dotenv
import pandas as pd
import numpy as np

import RICE

load_dotenv()


def test_dataframe_one_y():
    features = pd.DataFrame(columns=["X1", "X2", "X3"],
                            index=pd.date_range(start="20200101",
                                                end="20200107"),
                            data=[[0, np.nan, np.nan],
                                  [0, np.nan, np.nan],
                                  [2, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan],
                                  [4, np.nan, 4],
                                  [5, np.nan, 5],
                                  [2, np.nan, 3]])

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=features.index,
                                             data=[0, 0, 0, 0, 0, 0, 1]))


def test_dataframe_one_y_missing_x3_fillna():
    features = pd.DataFrame(columns=["X1", "X2"],
                            index=pd.date_range(start="20200101",
                                                end="20200107"),
                            data=[[0, np.nan],
                                  [0, np.nan],
                                  [2, np.nan],
                                  [np.nan, np.nan],
                                  [4, np.nan],
                                  [5, np.nan],
                                  [2, np.nan]])

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=features.index,
                                             data=[0, 0, 0, 0, 0, 0, 0]))


def test_dataframe_one_y_missing_x3_filltrue():
    features = pd.DataFrame(columns=["X1", "X2"],
                            index=pd.date_range(start="20200101",
                                                end="20200107"),
                            data=[[0, np.nan],
                                  [0, np.nan],
                                  [2, np.nan],
                                  [np.nan, np.nan],
                                  [4, np.nan],
                                  [5, np.nan],
                                  [2, np.nan]])

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features, missing_feature='filltrue')
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=features.index,
                                             data=[0, 0, 1, 0, 0, 0, 1]))


def test_dataframe_multiple_ys_y_columns():
    ys = ["Y1", "Y2"]
    xs = ["X1", "X2", "X3"]
    dates = pd.date_range(start="20200101", end="20200104").to_list()
    features = pd.DataFrame(index=pd.MultiIndex.from_product([dates, xs]),
                            columns=ys, data=[[0, 0],  # X1 0101
                                              [2, 2],  # X2
                                              [0, 0],  # X3
                                              [0, 0],  # X1 0102
                                              [2, 2],  # X2
                                              [0, 0],  # X3
                                              [2, 2],  # X1 0103
                                              [2, 2],  # X2
                                              [3, 3],  # X3
                                              [4, 2],  # X1 0104
                                              [2, 2],  # X2
                                              [4, 3]])  # X3

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=av.index,
                                             data=[0, 0, 0, 0, 1, 1, 0, 1]))


def test_dataframe_multiple_ys_x_columns():
    ys = ["Y1", "Y2"]
    xs = ["X1", "X2", "X3"]
    dates = pd.date_range(start="20200101", end="20200104").to_list()
    features = pd.DataFrame(index=pd.MultiIndex.from_product([dates, ys]),
                            columns=xs, data=[[0, 2, 0],
                                              [0, 2, 0],
                                              [0, 2, 0],
                                              [0, 2, 0],
                                              [2, 2, 3],
                                              [2, 2, 3],
                                              [4, 2, 4],
                                              [2, 2, 3]])

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=av.index,
                                             data=[0, 0, 0, 0, 1, 1, 0, 1]))


def test_dataframe_multiple_ys_x_columns_swapped():
    ys = ["Y1", "Y2"]
    xs = ["X1", "X2", "X3"]
    dates = pd.date_range(start="20200101", end="20200104").to_list()
    features = pd.DataFrame(index=pd.MultiIndex.from_product([ys, dates]),
                            columns=xs, data=[[0, 2, 0],
                                              [0, 2, 0],
                                              [0, 2, 0],
                                              [0, 2, 0],
                                              [2, 2, 3],
                                              [2, 2, 3],
                                              [4, 2, 4],
                                              [2, 2, 3]])

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=av.index,
                                             data=[0, 1, 0, 1, 0, 0, 0, 1]))


def test_dataframe_multiple_ys_y_columns_swapped():
    ys = ["Y1", "Y2"]
    xs = ["X1", "X2", "X3"]
    dates = pd.date_range(start="20200101", end="20200104").to_list()
    features = pd.DataFrame(index=pd.MultiIndex.from_product([xs, dates]),
                            columns=ys, data=[[0, 0],  # 0101 X1
                                              [0, 0],  # 0102
                                              [2, 2],  # 0103
                                              [4, 2],  # 0104
                                              [2, 2],  # 0101 X2
                                              [2, 2],  # 0102
                                              [2, 2],  # 0103
                                              [2, 2],  # 0104
                                              [0, 0],  # 0101 X3
                                              [0, 0],  # 0102
                                              [3, 3],  # 0103
                                              [4, 3]])  # 0104

    condition = RICE.RuleConditions(features_names=["X1", "X3"],
                                    features_index=[0, 2],
                                    bmin=[1, 3],
                                    bmax=[2, 4],
                                    xmin=[0, 0],
                                    xmax=[5, 5])

    av = condition.transform(features)
    print("")
    print("conditions: X1 in [1,2] and X3 in [3,4]")
    print("features:")
    print(features)
    print("Activation:")
    print(av)

    pd.testing.assert_series_equal(av,
                                   pd.Series(index=av.index,
                                             data=[0, 0, 0, 0, 1, 1, 0, 1]))


test_dataframe_one_y()
test_dataframe_one_y_missing_x3_fillna()
test_dataframe_one_y_missing_x3_filltrue()
test_dataframe_multiple_ys_x_columns()
test_dataframe_multiple_ys_x_columns_swapped()
test_dataframe_multiple_ys_y_columns()
test_dataframe_multiple_ys_y_columns_swapped()
