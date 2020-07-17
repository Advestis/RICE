# -*- coding: utf-8 -*-
"""
Created on 22 sept. 2016
@author: VMargot
"""
import copy
import operator
import functools

import numpy as np
import pandas as pd
import scipy.spatial.distance as scipy_dist
import matplotlib.pyplot as plt
import seaborn as sns

from bisect import bisect
from collections import Counter
from joblib import Parallel, delayed
from matplotlib import patches
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from typing import Union, List
from adutils import TransparentPath


def convert_string_to_list(thestring, dtype="str"):
    thestring = thestring.replace("[", "").replace("]", "").replace(" ", "")
    thestring = thestring.replace('"', "").replace("'", "").split(",")
    if dtype == "float":
        thestring = [float(i) for i in thestring]
    elif dtype == "int":
        thestring = [int(i) for i in thestring]
    return thestring


"""
---------
RuleConditions
---------
"""


class RuleConditions(object):
    """
    Class for binary rule condition
    """

    def __init__(
        self,
        features_names,
        features_index=None,
        bmin=None,
        bmax=None,
        xmin=None,
        xmax=None,
        values=None,
    ):

        assert isinstance(features_names, (tuple, list, np.ndarray)), (
            "Type of parameter must be iterable tuple, list or array"
            % features_names
        )
        self.features_names = features_names
        length = len(features_names)

        if features_index is not None:
            assert isinstance(features_index, (tuple, list, np.ndarray)), (
                "Type of parameter must be iterable tuple, list or array"
                % features_names
            )
            assert len(features_index) == length, (
                "Parameters must have the same length" % features_names
            )
        self.features_index = features_index

        if bmin is None:
            raise TypeError("bmin can not be None")
        if bmax is None:
            raise TypeError("bmax can not be None")

        assert isinstance(bmin, (tuple, list, np.ndarray)), (
            "Type of parameter must be iterable tuple, list or array"
            % features_names
        )
        assert len(bmin) == length, (
            "Parameters must have the same length" % features_names
        )
        assert isinstance(bmax, (tuple, list, np.ndarray)), (
            "Type of parameter must be iterable tuple, list or array"
            % features_names
        )
        assert len(bmax) == length, (
            "Parameters must have the same length" % features_names
        )
        if type(bmin[0]) != str:
            assert all(map(lambda a, b: a <= b, bmin, bmax)), (
                "Bmin must be smaller or equal than bmax (%s)" % features_names
            )
        self.bmin = bmin
        self.bmax = bmax
        if xmax is not None:
            assert isinstance(xmax, (tuple, list, np.ndarray)), (
                "Type of parameter must be iterable tuple, list or array"
                % features_names
            )
            assert len(xmax) == length, (
                "Parameters must have the same length" % features_names
            )
        if xmin is not None:
            assert isinstance(xmin, (tuple, list, np.ndarray)), (
                "Type of parameter must be iterable tuple, list or array"
                % features_names
            )
            assert len(xmin) == length, (
                "Parameters must have the same length" % features_names
            )
        self.xmin = xmin
        self.xmax = xmax

        if values is None:
            values = []
        else:
            assert isinstance(values, (tuple, list, np.ndarray)), (
                "Type of parameter must be"
                "iterable tuple, list or array" % features_names
            )

        self.values = [values]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = self.features_names
        return "Var: %s, Bmin: %s, Bmax: %s" % (features, self.bmin, self.bmax)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        to_hash = [
            (
                self.features_index[i],
                self.features_names[i],
                self.bmin[i],
                self.bmax[i],
            )
            for i in range(len(self.features_index))
        ]
        to_hash = frozenset(to_hash)
        return hash(to_hash)

    def check_index(self, features: pd.DataFrame,
                    missing_feature: str = 'fillna'):
        """ Check if content of ind matches content of self.features_names.
        Order does not matter.

        Parameters
        ----------
        features: pd.DataFrame
            shape=[n, d]

        missing_feature: str
            What to do in case some elements in self.features_names are not in
            x. Can be 'raise', 'fillna' or 'filltrue'. If 'raise',
            any missing feature in ind will raise an error. If 'fillna',
            any missnig feature will be filled with NaN, not activating the
            condition. If 'filltrue', will fill with a value that will always
            activated the condition.

        Returns
        -------
        None

        """

        l_ind = list(features)[:]
        l_feat = self.features_names[:]
        l_ind.sort()
        l_feat.sort()
        no_match = [f for f in l_feat if f not in l_ind]

        if len(no_match) > 0:
            if missing_feature == 'raise':
                to_raise = (
                    "Index and features do not match:"
                    f"\n  {no_match} are in condition's features but not in"
                    f" x's index \n(index is {l_ind})"
                    f"\n  condition's features are {l_feat}"
                )
                raise ValueError(to_raise)
            elif missing_feature == 'fillna':
                for mf in no_match:
                    features[mf] = pd.Series(index=features.index, data=np.nan)
            elif missing_feature == 'filltrue':
                print("chien")
                for mf in no_match:
                    bmin = self.bmin[self.features_names.index(mf)]
                    bmax = self.bmax[self.features_names.index(mf)]
                    # set to a value between bmin and bmax to be sure it
                    # activates
                    value = (bmin + bmax) / 2
                    features[mf] = pd.Series(index=features.index, data=value)
            else:
                raise ValueError(f"Unknown value {missing_feature} for "
                                 f"missing_features.")
        return features

    def order_columns(self, x: pd.DataFrame, missing_feature: str = 'fillna') \
            -> pd.DataFrame:
        """ Order x's columns according to features_namess, assuming x's
        columns are features. If x contains more features than
        self.features_names, returns the subset of x matching
        self.features_names, with columns ordered according to
        self.features_names.

        Parameters
        ----------
        x: pd.DataFrame
            x data to sort the columns of. shape=[n, d], so index is date 
            and columns is features names

        missing_feature: str
            What to do in case some elements in self.features_names are not in
            x. Can be 'raise', 'fillna' or 'filltrue'. If 'raise',
            any missing feature in ind will raise an error. If 'fillna',
            any missnig feature will be filled with NaN, not activating the
            condition. If 'filltrue', will fill with a value that will always
            activated the condition.

        Returns
        -------
        pd.DataFrame
            The DataFrame with columns matching self.features_names
        """

        self.check_index(x, missing_feature)
        x = x.loc[:, self.features_names]
        return x

    def tag_columns(self, x: pd.DataFrame) -> str:
        """Return 'y' if columns are Ys, 'x' if columns are features,
        raises IndexError if features are found nowhere.

        Parameters
        ----------
            x: pd.DataFrame
                The features Dataframe. shape=[n, d, m] or [n, m, d] or
                [d, n, m] of [m, n, d]. Has to be multiindexed.

        Returns
        -------
            str

        """
        for item in x.columns:
            if item in self.features_names:
                return 'x'
        for level in range(len(x.index.levels)):
            for item in list(x.index.levels[level]):
                if item in self.features_names:
                    return 'y'

        raise IndexError(f"None of the features in the condition"
                         f" were found in x")

    def transform(self, x: Union[np.ndarray, pd.Series, pd.DataFrame],
                  missing_feature: str = "fillna", columns: str = None)\
            -> Union[np.array, pd.Series]:
        """ Transforms a matrix xmat into an activation vector.

        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        x: Union[np.ndarray, pd.Series, pd.DataFrame]
            Assuming n=number of dates, d number of features and m number of
            Ys:
            Input data. shape=[n, d] or [n, d, m] or [n, m, d] or [d, n, m]
            of [m, n, d].

        missing_feature: str
            What to do in case some elements in self.features_names are not in
            x. Can be 'raise', 'fillna' or 'filltrue'. If 'raise',
            any missing feature in ind will raise an error. If 'fillna',
            any missnig feature will be filled with NaN, not activating the
            condition. If 'filltrue', will fill with a value that will always
            activated the condition.

        columns: str
            If 'y', assumes columns are y names and index second level is
            features names. If 'x', assumes the opposite, If None, will try
            to guess by looking for feature names in x (Default value = None).

        Returns
        -------
        activation_vector: Union[np.array, pd.Series]
            The activation vector, shape=n or shape=[n, m] if x is DataFrame.
            Multi-Indexed in that case.
        """

        if isinstance(x, np.ndarray):
            # Return is a np.array of shape n
            return self.transform_array(x)
        elif isinstance(x, pd.Series):
            # Return is a Series of shape n
            return self.transform_series(x)
        elif isinstance(x, pd.DataFrame):
            if not isinstance(x.index, pd.MultiIndex):
                return self.transform_series(x.stack(dropna=False),
                                             missing_feature)
            # Tries to guess if features name are in the columns. By
            # default, the y names are assumed to be the columns.
            if columns is None:
                columns = self.tag_columns(x)
            if columns != 'y' and columns != 'x':
                raise ValueError(f"Unknown value {columns} for columns.")

            # Return is a Multi-indexed Series of shape [n, m] (n dates and
            # m Ys)
            return self.transform_dataframe(x, columns, missing_feature)
        else:
            raise ValueError(f"Unknown type for x: {type(x)}")

    def transform_array(self, x: np.ndarray) -> np.array:
        """ Transforms a matrix into an activation vector.

        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise. The input x must have the same columns as
        the x used to create the condition. Indeed, The features on which
        the condition applies now their index position in the initial x, and
        those positions are used to find the correct column in the x passed as
        argument.

        Parameters
        ----------
        x: np.ndarray
            Input data containing ALL features. shape=[n, d]

        Returns
        -------
        activation_vector: np.array
            The activation vector, shape=[n]
        """

        length = len(self.features_names)
        x_cols = [x[:, self.features_index[i]] for i in range(length)]
        return self.transform_subarray(x_cols)

    def transform_subarray(
        self, x_cols: Union[pd.DataFrame, List]
    ) -> Union[pd.Series, np.ndarray]:
        """ Transforms a matrix into an activation vector.

        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise. The input x's columns must match
        self.features_namess if x is a DataFrame. If it is an array,
        it should have len(self.features_names) columns.

        Parameters
        ----------
        x_cols: Union[pd.DataFrame, List]
            Input data containing ALL features.
            shape=[len(self.features_names), d]

        Returns
        -------
        activation_vector: Union[pd.Series, np.array]
            The activation vector, shape=[n]
        """

        geq_min = True
        leq_min = True
        not_nan = True
        for i in range(len(self.features_names)):
            if isinstance(x_cols, pd.DataFrame):
                x_col = x_cols.iloc[:, i].values
            else:
                x_col = x_cols[i]
            # Turn x_col to array
            if len(x_col) > 1:
                x_col = np.squeeze(np.asarray(x_col))

            if type(self.bmin[i]) == str:
                x_col = np.array(x_col, dtype=np.str)

                temp = x_col == self.bmin[i]
                temp |= x_col == self.bmax[i]
                geq_min &= temp
                leq_min &= True
                not_nan &= True
            else:
                x_col = np.array(x_col, dtype=np.float)

                x_temp = [self.bmin[i] - 1 if x != x else x for x in x_col]
                geq_min &= np.greater_equal(x_temp, self.bmin[i])

                x_temp = [self.bmax[i] + 1 if x != x else x for x in x_col]
                leq_min &= np.less_equal(x_temp, self.bmax[i])

                not_nan &= np.isfinite(x_col)

        activation_vector = 1 * (geq_min & leq_min & not_nan)
        if isinstance(x_cols, pd.DataFrame):
            activation_vector = pd.Series(
                data=activation_vector, index=x_cols.index
            )

        return activation_vector

    def transform_series(self, x: pd.Series,
                         missing_feature: str = 'fillna') -> pd.Series:
        """ Transforms a pd.Series into an activation vector.

        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise. When unstacked, x can have any number of
        columns. Only the columns with names matching self.features_names
        will be kept.

        Parameters
        ----------
        x: pd.Series
            Input data. Multi-indexed. shape=[n, d] when unstacked.

        missing_feature: str
            What to do in case some elements in self.features_names are not in
            x. Can be 'raise', 'fillna' or 'filltrue'. If 'raise',
            any missing feature in ind will raise an error. If 'fillna',
            any missnig feature will be filled with NaN, not activating the
            condition. If 'filltrue', will fill with a value that will always
            activated the condition.

        Returns
        -------
        activation_vector: pd.Series
            The activation vector, shape=n
        """

        x_data = x.unstack()
        x_data = self.order_columns(x_data, missing_feature)
        return pd.Series(
            index=list(x_data.index), data=self.transform_subarray(x_data),
        )

    def transform_dataframe(self, x: pd.DataFrame, columns: str,
                            missing_feature: str = 'fillna') -> pd.Series:
        """ Transforms a pd.DataFrame into an activation vector.

        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        x: pd.DataFrame Input data.
            Multi-indexed. Index shape=[n, d] or [d, n], columns shape=[m], so
            multiindexed by date-features names or features names-date,
            and columns are y names, or Index shape=[m, d] or [d, m],
            columns shape=[n], so multiindexed by date-y names or y
            names-dates, and columns are features names. If the former is
            true, pass also columns='x'.

        columns: str
            if 'y', assumes columns are y names and index second level is
            features names. If 'x', assumes the opposite.

        missing_feature: str
            What to do in case some elements in self.features_names are not in
            x. Can be 'raise', 'fillna' or 'filltrue'. If 'raise',
            any missing feature in ind will raise an error. If 'fillna',
            any missnig feature will be filled with NaN, not activating the
            condition. If 'filltrue', will fill with a value that will always
            activated the condition.

        Returns
        -------
        activation_vector: pd.Series
            The activation vector. Index shape=[n, m]. So indexed by dates-y
            names.

        """

        date_index = 0
        other_index = 1
        if not isinstance(x.index.levels[0], pd.DatetimeIndex):
            date_index = 1
            other_index = 0
            if not isinstance(x.index.levels[1], pd.DatetimeIndex):
                raise IndexError("Could not find dates in MultiIndex!")

        dates = x.index.levels[date_index].to_list()
        dates.sort()
        if columns != "x" and columns != "y":
            raise ValueError(
                "Argument 'columns' must be 'x', 'y' {columns}"
            )
        if columns == "y":
            # columns are ynames, index are dates
            activation_vector = pd.DataFrame(
                index=dates, columns=x.columns, data=True
            )
            for yname in activation_vector.columns:
                data_x = x[yname]
                # If multiindex was x-date, we need to swap to end up with
                # date-x
                if other_index == 0:
                    data_x = data_x.unstack().T.stack(dropna=False)
                av = self.transform_series(data_x, missing_feature)
                activation_vector[yname] = av
        else:
            # Do not use set for it messes up columns ordering
            ynames = x.index.levels[other_index]
            # columns are ynames, index are dates
            activation_vector = pd.DataFrame(
                index=dates, columns=ynames, data=True
            )
            for yname in activation_vector.columns:
                idx = pd.IndexSlice
                if other_index == 1:
                    data_x = x.loc[idx[:, yname], :].droplevel(1).stack(
                        dropna=False
                    )
                else:
                    data_x = x.loc[idx[yname, :], :].droplevel(0).stack(
                        dropna=False
                    )
                av = self.transform_series(data_x, missing_feature)
                activation_vector[yname] = av
        return activation_vector.stack(dropna=False)

    """------   Getters   -----"""

    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, "Must be a string"

        return getattr(self, param)

    def get_attr(self):
        """
        To get a list of attributes of self.
        It is useful to quickly create a RuleConditions
        from intersection of two rules
        """
        return [
            self.features_names,
            self.features_index,
            self.bmin,
            self.bmax,
            self.xmin,
            self.xmax,
        ]

    """------   Setters   -----"""

    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


"""
---------
Rule
---------
"""


class Rule(object):
    """
    Class for a rule with a binary rule condition
    """

    def __init__(self, rule_conditions):

        assert (
            rule_conditions.__class__ == RuleConditions
        ), "Must be a RuleCondition object"

        self.conditions = rule_conditions
        self.length = len(rule_conditions.get_param("features_names"))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.conditions == other.conditions

    def __gt__(self, val):
        return self.get_param("pred") > val

    def __lt__(self, val):
        return self.get_param("pred") < val

    def __ge__(self, val):
        return self.get_param("pred") >= val

    def __le__(self, val):
        return self.get_param("pred") <= val

    def __str__(self):
        return "rule: " + self.conditions.__str__()

    def __hash__(self):
        return hash(self.conditions)

    def pretty_print(self):
        s = (
            f"{self.get_param('name')} ("
            f"pred={str(round(self.get_param('prediction'), 2))})\\\\"
        )
        for x, bmin, bmax in zip(
            self.conditions.features_names,
            self.conditions.bmin,
            self.conditions.bmax,
        ):
            s += f"${x} \\in ]{str(bmin)},{str(bmax)}]$ \\\\"
        return s

    def test_included(self, rule, x=None):
        """
        Test to know if a rule (self) and an other (rule)
        are included
        """

        # No need to check x type, it is done in get_activation()

        activation_self = self.get_activation(x)
        activation_other = rule.get_activation(x)

        intersection = np.logical_and(activation_self, activation_other)

        if np.allclose(intersection, activation_self) or np.allclose(
            intersection, activation_other
        ):
            return None
        else:
            return 1 * intersection

    def test_variables(self, rule):
        """
        Test to know if a rule (self) and an other (rule)
        have conditions on the same features.
        """
        c1 = self.conditions
        c2 = rule.conditions

        c1_name = c1.get_param("features_names")
        c2_name = c2.get_param("features_names")
        if len(set(c1_name).intersection(c2_name)) != 0:
            return True
        else:
            return False

    def test_length(self, rule, length):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected to have a new rule of length length.
        """
        return self.get_param("length") + rule.get_param("length") == length

    def intersect_test(self, rule, x):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected.

        Test 1: the sum of complexities of self and rule are egal to l
        Test 2: self and rule have not condition on the same variable
        Test 3: self and rule have not included activation
        """

        # No need to check x type, it is done in test_included()

        if self.test_variables(rule) is False:
            return self.test_included(rule=rule, x=x)
        else:
            return None

    def union_test(self, activation, gamma=0.80, x=None):
        # TODO use in IMORA
        """
        Test to know if a rule (self) and an activation vector have
        at more gamma percent of points in common

        If True, to add to ruleset. Else, discard.
        """

        # No need to check X type, it is done in get_activation()

        self_vect = self.get_activation(x)
        intersect_vect = np.logical_and(self_vect, activation)

        pts_inter = np.sum(intersect_vect)
        pts_rule = np.sum(activation)
        pts_self = np.sum(self_vect)

        ans = (pts_inter < gamma * pts_self) and (pts_inter < gamma * pts_rule)

        return ans

    def intersect_conditions(self, rule):
        """
        Compute an RuleCondition object from the intersection of an rule
        (self) and an other (rulessert)
        """
        conditions_1 = self.conditions
        conditions_2 = rule.conditions

        conditions = list(
            map(
                lambda c1, c2: c1 + c2,
                conditions_1.get_attr(),
                conditions_2.get_attr(),
            )
        )

        return conditions

    def intersect(self, rule, cov_min, cov_max, x, low_memory):
        """
        Compute a suitable rule object from the intersection of an rule
        (self) and an other (rule).
        Suitable means that self and rule satisfied the intersection test
        """

        # No need to check x type, it is done in intersect_test()

        new_rule = None
        # if self.get_param('pred') * rule.get_param('pred') > 0:
        activation = self.intersect_test(rule, x)
        if activation is not None:
            cov = calc_coverage(activation)
            if cov_min <= cov <= cov_max:
                conditions_list = self.intersect_conditions(rule)

                new_conditions = RuleConditions(
                    features_names=conditions_list[0],
                    features_index=conditions_list[1],
                    bmin=conditions_list[2],
                    bmax=conditions_list[3],
                    xmax=conditions_list[5],
                    xmin=conditions_list[4],
                )
                new_rule = Rule(new_conditions)
                if low_memory is False:
                    new_rule.set_params(activation=activation)

        return new_rule

    def calc_stats(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        y: Union[np.ndarray, pd.DataFrame, pd.Series],
        method: str = "mse",
        cov_min: float = 0.01,
        cov_max: float = 0.5,
        low_memory: bool = False,
        first_selection: bool = False,
    ):
        """
        Calculation of all statistics of an rules

        Parameters
        ----------

        x: Union[np.ndarray, pd.DataFrame, pd.Series]
            The training input samples, shape = [n, d].

        y: Union[np.ndarray, pd.DataFrame, pd.Series]
            The normalized target values, shape = [n].

        method: str
            The method mse_function or msecriterion. Default "mse".

        cov_min: float
            The minimal coverage of one rule. Default 0.01.

        cov_max: float
            The maximal coverage of one rule. Default 0.5.

        low_memory: bool
            To save activation vectors of rules. Default False.

        first_selection: bool
            If True, computes activation vector and
            coverage, check cov conditions and returns. If False, computes all
            other stats too (prediction, prediction vec, and mse). Will call
            itself with first_selection True if calc_stats had not been called
            already on this rule with first_selection True.

        Return
        ------
        None
        """

        dtype = y.dtype
        if first_selection:
            self.set_params(first_selected=True)
            self.set_params(out=False)
            # activation_vector is a np.array or a pd.Series, possibly
            # multi-indexed
            activation_vector = self.calc_activation(x=x)

            if sum(activation_vector) <= 0:
                self.set_params(out=True)
                return

            if low_memory is False:
                self.set_params(activation=activation_vector)

            cov = calc_coverage(activation_vector)
            self.set_params(cov=cov)

            if cov >= cov_max or cov <= cov_min:
                self.set_params(out=True)
            return
        else:
            if not hasattr(self, "first_selected"):
                self.calc_stats(
                    x,
                    y,
                    method,
                    cov_min,
                    cov_max,
                    low_memory,
                    first_selection=True,
                )
                if self.get_param("out"):
                    return

            if (
                low_memory is True
            ):  # In that case, calling with first_selection True did not
                # save activation vector.
                activation_vector = self.calc_activation(x)
            else:
                activation_vector = self.get_param("activation")

            # floats
            prediction = calc_prediction(activation_vector, y)
            self.set_params(pred=prediction)
            complementary_prediction = calc_prediction(
                1 - activation_vector, y
            )

            cond_var = calc_variance(activation_vector, y)
            self.set_params(var=cond_var)

            # If Ys are binned, their values are integer, but means will be
            # floats Round then cast predictions to integer in that case.
            if dtype == int:
                prediction = int(np.round(prediction, 0))
                complementary_prediction = int(
                    np.round(complementary_prediction, 0)
                )

            prediction_vector = activation_vector * prediction

            if isinstance(complementary_prediction, np.array):
                np.place(
                    prediction_vector,
                    prediction_vector == 0,
                    complementary_prediction,
                )
            else:
                prediction_vector.loc[
                    prediction_vector == 0
                ] = complementary_prediction

            # The final prediction vector is as follows : At each Ys
            # activated by the rule, is the mean of all those Ys. At each Ys
            # NOT activated by the rule, is the mean of all those Ys.

            # Criterion is an evaluation of the goodness of prediction
            rez = calc_criterion(prediction_vector, y, method)
            self.set_params(crit=rez)
            exec("self.set_params(" + method + "=rez)")

    def calc_activation(self, x=None):
        """ Compute the activation vector of a Rule object
        """
        # No need to check x type, it is done in transform()
        return self.conditions.transform(x)

    def predict(self, x: Union[np.array, pd.Series, pd.DataFrame] = None):
        """ Compute the prediction of an rule

        Parameters
        ----------
        x : Union[np.array, pd.Series, pd.DataFrame]
            Features. Shape = [n, d] or shape = [n, d, m]
            (can be Multi-indexed) (Default value = None)

        Returns
        -------
        np.array or pd.Series
            Prediction vector. Shape = [n, d] (can be Multi-indexed)

        """
        # No need to check x type, it is done in calc_activation()
        prediction = self.get_param("pred")
        if x is not None:
            activation = self.calc_activation(x=x)
        else:
            activation = self.get_activation()

        return prediction * activation

    def score(self, x, y, sample_weight=None, score_type="Rate"):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x: Union[array-like, pd.DataFrame, pd.Series]
            shape = (n_samples, n_features). Test samples.

        y: Union[array-like, pd.DataFrame, pd.Series]
            shape = (n_samples) or (n_samples, n_outputs).
            True values for X.

        sample_weight: array-like
            shape = [n_samples], optional
            Sample weights.

        score_type: str

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.

            or

        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """
        # No need to check x type, it is done in predict()
        prediction_vector = self.predict(x)

        data_y = y
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            data_y = y.values

        data_y = np.extract(np.isfinite(data_y), data_y)
        prediction_vector = np.extract(np.isfinite(data_y), prediction_vector)

        if score_type == "Classification":
            th_val = (min(y) + max(y)) / 2.0
            prediction_vector = list(
                map(
                    lambda p: min(y) if p < th_val else max(y),
                    prediction_vector,
                )
            )
            return accuracy_score(y, prediction_vector)

        elif score_type == "Regression":
            return r2_score(
                y,
                prediction_vector,
                sample_weight=sample_weight,
                multioutput="variance_weighted",
            )

    def make_name(self, num: int, learning: "Learning" = None):
        """
        Add an attribute name to self

        Parameters
        ----------
        num : int
            index of the rule in an ruleset

        learning : Learning
            If leaning is not None the name of self will
            be defined with the name of learning.
            Default None
        """
        name = f"R{str(num)}"
        length = self.get_param("length")
        name += f"({str(length)})"
        prediction = self.get_param("pred")
        if prediction > 0:
            name += "+"
        elif prediction < 0:
            name += "-"

        if learning is not None:
            dtstart = learning.get_param("dtstart")
            dtend = learning.get_param("dtend")
            if dtstart is not None:
                name += str(dtstart) + " "
            if dtend is not None:
                name += str(dtend)

        self.set_params(name=name)

    """------   Getters   -----"""

    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, "Must be a string"
        assert hasattr(self, param), "self.%s must be calculate before" % param
        return getattr(self, param)

    def get_activation(self, x=None):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        # No need to check x type, it is done in transform()
        if x is not None:
            return self.conditions.transform(x)
        else:
            if hasattr(self, "activation"):
                return self.get_param("activation")
            else:
                print("No activation vector for %s" % str(self))
            return None

    def get_predictions_vector(self, x=None):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        # No need to check x type, it is done in calc_activation()
        if hasattr(self, "pred"):
            prediction = self.get_param("pred")
            if hasattr(self, "activation"):
                return prediction * self.get_param("activation")
            else:
                return prediction * self.calc_activation(x)
        else:
            return None

    """------   Setters   -----"""

    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


"""
---------
RuleSet
---------
"""


class RuleSet(object):
    """
    Class for a ruleset. It's a kind of list of rule object
    """

    def __init__(self, rs=None, path: TransparentPath = None):
        self.rules_df = None
        self.rules = None
        if path is None:
            if rs is None:
                raise ValueError(
                    "RuleSet: need to provide either rs or "
                    "path. If you want an empty RuleSet, "
                    "you can pass rs=[]."
                )
            self.activation = None
            if type(rs) in [list, np.ndarray]:
                self.rules = rs
            elif type(rs) == RuleSet:
                self.rules = rs.get_rules()
            self.make_rules_df()
        else:
            self.load_rules_from_file(path)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "ruleset: %s rules" % str(len(self.rules))

    def __gt__(self, val):
        return [rule > val for rule in self.rules]

    def __lt__(self, val):
        return [rule < val for rule in self.rules]

    def __ge__(self, val):
        return [rule >= val for rule in self.rules]

    def __le__(self, val):
        return [rule <= val for rule in self.rules]

    def __add__(self, ruleset):
        return self.extend(ruleset)

    def __getitem__(self, i):
        return self.get_rules()[i]

    def __len__(self):
        try:
            return len(self.get_rules())
        except TypeError:
            return 0

    def __del__(self):
        if len(self) > 0:
            nb_rules = len(self)
            i = 0
            while i < nb_rules:
                del self[0]
                i += 1

    def __delitem__(self, rules_id):
        del self.rules[rules_id]

    def load_rules_from_file(self, path: TransparentPath):
        if not path.is_file():
            raise FileNotFoundError(f"Could not find ruleset file {path}.")

        self.rules_df = path.read(index_col=0)
        cols = [c.lower() for c in self.rules_df.columns]
        self.rules_df.columns = cols

        if not isinstance(self.rules_df, pd.DataFrame):
            raise TypeError(
                "Expected a DataFrame in the input ruleset file,"
                f"got {type(self.rules_df)} instead."
            )

        expected_columns = ["features_names", "bmin", "bmax"]
        try:
            self.rules_df.loc[:, expected_columns]
        except KeyError:
            raise KeyError(
                "The DataFrame in the input file must contain "
                "at least the following columns:\n"
                f"{expected_columns}"
            )

        rules = []
        for rule_name in self.rules_df.index:
            fi = None
            if "features_index" in self.rules_df.columns:
                fi = self.rules_df.loc[rule_name, "features_index"]
            fn = convert_string_to_list(
                self.rules_df.loc[rule_name, "features_names"]
            )
            bmin = convert_string_to_list(
                self.rules_df.loc[rule_name, "bmin"], dtype="float"
            )
            bmax = convert_string_to_list(
                self.rules_df.loc[rule_name, "bmax"], dtype="float"
            )

            rc = RuleConditions(
                features_names=fn, features_index=fi, bmin=bmin, bmax=bmax,
            )
            rule = Rule(rc)
            rule.set_params(name=rule_name)
            sub_df = self.rules_df.loc[rule_name].drop(
                ["features_names", "bmin", "bmax"]
            )
            if "features_index" in self.rules_df.columns:
                sub_df.drop("features_index", inplace=True)
            for col in sub_df.index:
                attr_name = col.lower()
                exec(f"rule.set_params({attr_name}=sub_df[col])")
            rules.append(rule)
        self.rules = rules
    
    def make_rules_df(self):
        if len(self) == 0:
            return pd.DataFrame()
        columns = ["features_names", "bmin", "bmax"]
        cond = self.rules[0].__dict__["conditions"]
        if cond.features_index is not None:
            columns.append("features_index")
        if cond.xmax is not None:
            columns.append("xmax")
        if cond.xmin is not None:
            columns.append("xmin")

        to_skip = ["conditions", "name", "length"]
        for attr in self.rules[0].__dict__:
            if attr in to_skip:
                continue
            columns.append(attr)

        self.rules_df = pd.DataFrame(index=[r.name for r in self.rules],
                                     columns=columns)

        for rule in self.rules:
            name = rule.name
            self.rules_df.loc[name, "features_names"] = \
                rule.conditions.features_names
            self.rules_df.loc[name, "bmax"] = rule.conditions.bmax
            self.rules_df.loc[name, "bmin"] = rule.conditions.bmin
            if "features_index" in self.rules_df.columns:
                self.rules_df.loc[name, "features_index"] = \
                    rule.conditions.features_index
            if "xmax" in self.rules_df.columns:
                self.rules_df.loc[name, "xmax"] = rule.conditions.xmax
            if "xmin" in self.rules_df.columns:
                self.rules_df.loc[name, "xmin"] = rule.conditions.xmin
            for attr in rule.__dict__:
                if attr in to_skip:
                    continue
                exec(f"self.rules_df.loc[name, '{attr}'] = rule.{attr}")

    # noinspection PyUnresolvedReferences
    def save(self, path: Union[str, "pathlib.Path", TransparentPath],
             extension="csv", **kwargs):
        if not isinstance(path, TransparentPath):
            path = TransparentPath(path)
        if self.rules_df is None:
            self.make_rules_df()
        if path.is_dir():
            path = path / f"ruleset.{extension}"
        path.write(self.rules_df, **kwargs)

    def append(self, rule):
        """
        Add one rule to a RuleSet object (self).
        """
        assert rule.__class__ == Rule, "Must be a rule object (try extend)"
        if any(map(lambda r: rule == r, self)) is False:
            self.rules.append(rule)

    def extend(self, ruleset):
        """
        Add rules form a ruleset to a RuleSet object (self).
        """
        assert ruleset.__class__ == RuleSet, "Must be a ruleset object"
        "ruleset must have the same Learning object"
        rules_list = ruleset.get_rules()
        self.rules.extend(rules_list)
        return self

    def insert(self, idx, rule):
        """
        Insert one rule to a RuleSet object (self) at the position idx.
        """
        assert rule.__class__ == Rule, "Must be a rule object"
        self.rules.insert(idx, rule)

    def pop(self, idx=None):
        """
        Drop the rule at the position idx.
        """
        self.rules.pop(idx)

    def extract_greater(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        greater than val.
        """
        rules_list = list(
            filter(lambda rule: rule.get_param(param) > val, self)
        )
        return RuleSet(rules_list)

    def extract_least(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        least than val.
        """
        rules_list = list(
            filter(lambda rule: rule.get_param(param) < val, self)
        )
        return RuleSet(rules_list)

    def extract_length(self, length):
        """
        Extract a RuleSet object from self such as each rules have a
        length l.
        """
        rules_list = list(
            filter(lambda rule: rule.get_param("length") == length, self)
        )
        return RuleSet(rules_list)

    def extract(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        equal to val.
        """
        rules_list = list(
            filter(lambda rule: rule.get_param(param) == val, self)
        )
        return RuleSet(rules_list)

    def index(self, rule):
        """
        Get the index a rule in a RuleSet object (self).
        """
        assert rule.__class__ == Rule, "Must be a rule object"
        self.get_rules().index(rule)

    def replace(self, idx, rule):
        """
        Replace rule at position idx in a RuleSet object (self)
        by a new rule.
        """
        self.rules.pop(idx)
        self.rules.insert(idx, rule)

    def sort_by(self, crit, maximized):
        """
        Sort the RuleSet object (self) by a criteria criterion
        """
        self.rules = sorted(
            self.rules, key=lambda x: x.get_param(crit), reverse=maximized
        )

    def drop_duplicates(self):
        """
        Drop duplicates rules in RuleSet object (self)
        """
        rules_list = list(set(self.rules))
        return RuleSet(rules_list)

    def to_df(self, cols=None):
        """
        To transform an ruleset into a pandas DataFrame
        """
        if cols is None:
            cols = [
                "Features_Name",
                "BMin",
                "BMax",
                "Cov",
                "Pred",
                "Var",
                "Crit",
                "Significant",
            ]

        df = pd.DataFrame(index=self.get_rules_name(), columns=cols)

        for col_name in cols:
            att_name = col_name.lower()
            if all([hasattr(rule, att_name) for rule in self]):
                df[col_name] = [rule.get_param(att_name) for rule in self]

            elif all(
                [hasattr(rule.conditions, att_name.lower()) for rule in self]
            ):
                df[col_name] = [
                    rule.conditions.get_param(att_name) for rule in self
                ]

        return df

    def calc_pred(self, y_train, x_train=None, x_test=None):
        """
        Computes the prediction vector
        using an rule based partition
        """

        # TODO: maybe add tests on index if dataframes or series ?
        data_y = y_train
        data_x = x_train
        data_x_test = x_test
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            data_y = y_train.values
        if x_train is not None:
            if isinstance(x_train, pd.Series) or isinstance(
                x_train, pd.DataFrame
            ):
                data_x = x_train.values
        if x_test is not None:
            if isinstance(x_test, pd.Series) or isinstance(
                x_test, pd.DataFrame
            ):
                data_x_test = x_test.values

        # Activation of all rules in the learning set
        activation_matrix = np.array(
            [rule.get_activation(data_x) for rule in self]
        )

        if data_x_test is None:
            prediction_matrix = activation_matrix.T
        else:
            prediction_matrix = [
                rule.calc_activation(data_x_test) for rule in self
            ]
            prediction_matrix = np.array(prediction_matrix).T

        no_activation_matrix = np.logical_not(prediction_matrix)

        nb_rules_active = prediction_matrix.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

        # Activation of the intersection of all NOT activated rules at each row
        no_activation_vector = np.dot(no_activation_matrix, activation_matrix)
        no_activation_vector = np.array(no_activation_vector, dtype="int")

        dot_activation = np.dot(prediction_matrix, activation_matrix)
        dot_activation = np.array(
            [
                np.equal(act, nb_rules)
                for act, nb_rules in zip(dot_activation, nb_rules_active)
            ],
            dtype="int",
        )

        # Calculation of the binary vector for cells of the partition et
        # each row
        cells = (dot_activation - no_activation_vector) > 0

        # Calculation of the expectation of the complementary
        no_act = 1 - self.calc_activation(data_x)
        no_pred = np.mean(np.extract(data_y, no_act))

        # Get empty significant cells
        significant_list = np.array(
            self.get_rules_param("significant"), dtype=int
        )
        significant_rules = np.where(significant_list == 1)[0]
        temp = prediction_matrix[:, significant_rules]
        nb_rules_active = temp.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1
        empty_cells = np.where(nb_rules_active == -1)[0]

        # Get empty insignificant cells
        bad_cells = np.where(np.sum(cells, axis=1) == 0)[0]
        bad_cells = list(filter(lambda i: i not in empty_cells, bad_cells))

        # Calculation of the conditional expectation in each cell
        prediction_vector = [calc_prediction(act, data_y) for act in cells]
        prediction_vector = np.array(prediction_vector)

        prediction_vector[bad_cells] = no_pred
        prediction_vector[empty_cells] = 0.0

        return prediction_vector, bad_cells, empty_cells

    def calc_activation(
        self,
        x: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
        low_memory: bool = True,
    ):
        """ Compute the  activation vector of a set of rules

        Parameters
        ----------
        x: Union[np.ndarray, pd.Series, pd.DataFrame]
            An optionnal feature matrix on which the activation vector is
            computed. If not specified, use the already computed activation
            vectors or the rules. Shape=[n, d] or [n, d, m]. (Default value
            = None).

        low_memory: bool
            If False, set the result as class attribute.

        Returns
        -------
        activation_vectors : Union[np.ndarray, pd.Series]

        """
        # No need to check x type, it is done in get_activation
        activation_vectors = [rule.get_activation(x) for rule in self]
        if len(activation_vectors) == 0:
            return np.array([])
        if isinstance(activation_vectors[0], pd.Series):
            # Giving a list of series to pd.DataFrame will use the first
            # Serie index as columns and will put each Serie as a line. Then
            # sum will sum from top to bottom and give us bask a Serie with
            # the initial index
            activation_vectors = pd.DataFrame(data=activation_vectors).sum()
        else:
            activation_vectors = np.sum(activation_vectors, axis=0)
        activation_vectors = 1 * activation_vectors.astype("bool")

        if low_memory is False:
            self.set_params(activation=activation_vectors)

        return activation_vectors

    def calc_coverage(self, x=None):
        """
        Compute the coverage rate of a set of rules
        """
        # No need to check x type, it is done in calc_activation
        if len(self) > 0:
            activation_vector = self.calc_activation(x)
            cov = calc_coverage(activation_vector)
        else:
            cov = 0.0
        return cov

    def predict(self, y_train, x_train, x_test):
        """
        Computes the prediction vector for a given X and a given aggregation
        method
        """
        # No need to check x nor y type, it is done in calc_pred
        prediction_vector, bad_cells, no_rules = self.calc_pred(
            y_train, x_train, x_test
        )
        return prediction_vector, bad_cells, no_rules

    def make_rule_names(self):
        """
        Add an attribute name at each rule of self
        """
        list(
            map(
                lambda rule, rules_id: rule.make_name(rules_id),
                self,
                range(len(self)),
            )
        )

    def make_selected_df(self):
        df = self.to_df()

        df.rename(
            columns={
                "Cov": "Coverage",
                "Pred": "Prediction",
                "Var": "Variance",
                "Crit": "Criterion",
            },
            inplace=True,
        )

        df["Conditions"] = [make_condition(rule) for rule in self]
        selected_df = df[
            ["Conditions", "Coverage", "Prediction", "Variance", "Criterion"]
        ].copy()

        selected_df["Coverage"] = selected_df.Coverage.round(2)
        selected_df["Prediction"] = selected_df.Prediction.round(2)
        selected_df["Variance"] = selected_df.Variance.round(2)
        selected_df["Criterion"] = selected_df.Criterion.round(2)

        return selected_df

    def plot_counter_variables(self, nb_max=None):
        counter = get_variables_count(self)

        x_labels = list(map(lambda item: item[0], counter))
        values = list(map(lambda item: item[1], counter))

        f = plt.figure()
        ax = plt.subplot()

        if nb_max is not None:
            x_labels = x_labels[:nb_max]
            values = values[:nb_max]

        g = sns.barplot(y=x_labels, x=values, ax=ax, ci=None)
        g.set(xlim=(0, max(values) + 1), ylabel="Variable", xlabel="Count")

        return f

    def plot_dist(self, x=None, metric=None):

        # No need to check x type, it is done in get_predictions_vector()

        rules_names = self.get_rules_name()

        predictions_vector_list = [
            rule.get_predictions_vector(x) for rule in self
        ]
        predictions_matrix = np.array(predictions_vector_list)

        distance_vector = scipy_dist.pdist(predictions_matrix, metric=metric)
        distance_matrix = scipy_dist.squareform(distance_vector)

        # Set up the matplotlib figure
        f = plt.figure()
        ax = plt.subplot()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(distance_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        vmax = np.max(distance_matrix)
        vmin = np.min(distance_matrix)
        # center = np.mean(distance_matrix)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            distance_matrix,
            cmap=cmap,
            ax=ax,
            vmax=vmax,
            vmin=vmin,
            center=1.0,
            square=True,
            xticklabels=rules_names,
            yticklabels=rules_names,
            mask=mask,
        )

        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        return f

    """------   Getters   -----"""

    def get_candidates(self, x, k, length, method, nb_jobs):

        # No need to check x type, it is done in find_cluster()
        candidates = []
        for alength in [1, length - 1]:
            rs_length_l = self.extract_length(alength)
            if method == "cluter":
                if (
                    all(
                        map(lambda rule: hasattr(rule, "cluster"), rs_length_l)
                    )
                    is False
                ):
                    clusters = find_cluster(rs_length_l, x, k, nb_jobs)
                    self.set_rules_cluster(clusters, alength)

                rules_list = []
                for i in range(k):
                    sub_rs = rs_length_l.extract("cluster", i)
                    if len(sub_rs) > 0:
                        sub_rs.sort_by("var", True)
                        rules_list.append(sub_rs[0])

            elif method == "best":
                rs_length_l.sort_by("crit", False)
                rules_list = rs_length_l[:k]

            else:
                print(
                    "Choose a method among [cluster, best] to select candidat"
                )
                rules_list = rs_length_l.rules

            candidates.append(RuleSet(rules_list))

        return candidates[0], candidates[1]

    def get_rules_param(self, param):
        """
        To get the list of a parameter param of the rules in self
        """
        return [rule.get_param(param) for rule in self]

    def get_rules_name(self):
        """
        To get the list of the name of rules in self
        """
        try:
            return self.get_rules_param("name")
        except AssertionError:
            self.make_rule_names()
            return self.get_rules_param("name")

    def get_rules(self):
        """
        To get the list of rule in self
        """
        return self.rules

    def get_activation(self, x=None):
        """
        To get the activation vector of all rules in self
        """
        if self.activation is None or x is not None:
            self.calc_activation(x)
        return self.activation

    """------   Setters   -----"""

    def set_rules(self, rules_list):
        """
        To set a list of rule in self
        """
        assert type(rules_list) == list, "Must be a list object"
        self.rules = rules_list

    def set_rules_cluster(self, params, length):
        rules_list = list(
            filter(lambda rule: rule.get_param("length") == length, self)
        )
        list(
            map(
                lambda rule, rules_id: rule.set_params(
                    cluster=params[rules_id]
                ),
                rules_list,
                range(len(rules_list)),
            )
        )
        rules_list += list(
            filter(lambda rule: rule.get_param("length") != length, self)
        )

        self.rules = rules_list

    """------   Setters   -----"""

    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


"""
---------
Learning
---------
"""


class Learning(BaseEstimator):
    """
    ...
    """

    def __init__(self, **parameters):
        """

        Parameters
        ----------
        alpha : {float type such as 0 < th < 1/4} default 1/5
                The main parameter

        nb_bucket : {int type} default max(3, n^1/d) with n the number of row
                    and d the number of features
                    Choose the number a bucket for the discretization

        l_max : {int type} default d
                 Choose the maximal length of one rule

        gamma : {float type such as 0 <= gamma <= 1} default 1
                Choose the maximal intersection rate begin a rule and
                a current selected ruleset

        k : {int type} default 500
            The maximal number of candidate to increase length

        nb_jobs : {int type} default number of core -2
                  Select the number of lU used
        """
        self.selected_rs = RuleSet([])
        self.ruleset = RuleSet([])
        self.bins = dict()
        self.critlist = []
        self.low_memory = False
        self.k = 150
        self.method = "best"
        self.alpha = 1.0 / 2 - 1.0 / 100
        self.gamma = 0.95
        self.nb_jobs = -2
        self.coverage = True

        for arg, val in parameters.items():
            setattr(self, arg, val)

    def __str__(self):
        learning = "Learning"
        # learning = self.get_param('cpname') + ': '
        # learning += self.get_param('target')
        return learning

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def fit(self, x, y, features_names=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        x: Union[array-like, sparse matrix, pd.DataFrame, pd.Series]
            shape = [n, d]. The training input samples.

        y: Union[array-like, pd.DataFrame, pd.Series]
            shape = [n]. The target values (real numbers).

        features_names : List
            Optional. Name of each features.
        """

        # TODO : maybe check index if df or series ?
        data_x = x
        data_y = y
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            data_y = y.values
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            data_x = x.values

        # Check type for data
        data_x = check_array(data_x, dtype=None, force_all_finite=False)

        data_y = check_array(
            data_y, dtype=None, ensure_2d=False, force_all_finite=False
        )

        # Creation of data-driven parameters
        if hasattr(self, "beta") is False:
            beta = 1.0 / pow(data_x.shape[0], 1.0 / 4 - self.alpha / 2.0)
            self.set_params(beta=beta)

        if hasattr(self, "epsilon") is False:
            beta = self.get_param("beta")
            epsilon = beta * np.std(data_y)
            self.set_params(epsilon=epsilon)

        if hasattr(self, "covmin") is False:
            covmin = 1.0 / pow(data_x.shape[0], self.alpha)
            self.set_params(covmin=covmin)

        if hasattr(self, "nb_bucket") is False:
            nb_bucket = max(
                10, int(np.sqrt(pow(data_x.shape[0], 1.0 / data_x.shape[1])))
            )

            nb_bucket = min(nb_bucket, data_x.shape[0])
            self.set_params(nb_bucket=nb_bucket)

        if hasattr(self, "covmax") is False:
            covmax = 1.0
            self.set_params(covmax=covmax)

        if hasattr(self, "calcmethod") is False:
            if len(set(data_y)) > 2:
                # Binary classification case
                calcmethod = "mse"
            else:
                # Regression case
                calcmethod = "mae"
            self.set_params(calcmethod=calcmethod)

        features_index = range(data_x.shape[1])
        if features_names is None:
            features_names = ["X" + str(i) for i in features_index]

        self.set_params(features_index=features_index)
        self.set_params(features_names=features_names)

        if hasattr(self, "l_max") is False:
            l_max = len(features_names)
            self.set_params(l_max=l_max)

        # Turn the matrix X in a discret matrix
        x_discretized = self.discretize(data_x)
        self.set_params(X=x_discretized)

        # Normalization of y
        ymean = np.nanmean(data_y)
        ystd = np.nanstd(data_y)
        self.set_params(ymean=ymean)
        self.set_params(ystd=ystd)

        self.set_params(y=data_y)

        # looking for good rules
        self.find_rules()  # works in columns not in lines

        self.set_params(fitted=True)

    def find_rules(self):
        """
        Find all rules for all length <= l
        then selects the best subset by minimization
        of the empirical risk
        """
        l_max = self.get_param("l_max")
        assert l_max > 0, "l_max must be strictly superior to 0"

        selected_rs = self.get_param("selected_rs")

        # --------------
        # DESIGNING PART
        # --------------
        self.calc_length_1()
        ruleset = self.get_param("ruleset")

        if len(ruleset) > 0:
            for k in range(2, l_max + 1):
                print("Designing of rules of length %s" % str(k))
                if len(selected_rs.extract_length(k)) == 0:
                    # seeking a set of rules with a length l
                    ruleset_length_up = self.calc_length_c(k)

                    if len(ruleset_length_up) > 0:
                        ruleset += ruleset_length_up
                        self.set_params(ruleset=ruleset)
                    else:
                        print("No rule for length %s" % str(k))
                        break

                    ruleset.sort_by("crit", False)
            self.set_params(ruleset=ruleset)

            # --------------
            # SELECTION PART
            # --------------
            print("----- Selection ------")
            selected_rs = self.select_rules(0)

            ruleset.make_rule_names()
            self.set_params(ruleset=ruleset)
            selected_rs.make_rule_names()
            self.set_params(selected_rs=selected_rs)

        else:
            print("No rule found !")

    def calc_length_1(self):
        """
        Compute all rules of length one and keep the best.
        """
        features_names = self.get_param("features_names")
        features_index = self.get_param("features_index")
        x = self.get_param("x")
        calcmethod = self.get_param("calcmethod")
        y = self.get_param("y")
        cov_max = self.get_param("covmax")
        cov_min = self.get_param("covmin")
        low_memory = self.get_param("low_memory")

        jobs = min(len(features_names), self.get_param("nb_jobs"))

        if jobs == 1:
            ruleset = list(
                map(
                    lambda var, idx: make_rules(
                        var,
                        idx,
                        x,
                        y,
                        calcmethod,
                        cov_min,
                        cov_max,
                        low_memory,
                    ),
                    features_names,
                    features_index,
                )
            )
        else:
            ruleset = Parallel(n_jobs=jobs, backend="multiprocessing")(
                delayed(make_rules)(
                    var, idx, x, y, calcmethod, cov_min, cov_max, low_memory
                )
                for var, idx in zip(features_names, features_index)
            )

        ruleset = functools.reduce(operator.add, ruleset)

        ruleset = RuleSet(ruleset)
        ruleset.sort_by("crit", False)

        self.set_params(ruleset=ruleset)

    def calc_length_c(self, length):
        """
        Returns a ruleset of rules with a given length.
        """
        nb_jobs = self.get_param("nb_jobs")
        x = self.get_param("x")
        calcmethod = self.get_param("calcmethod")
        y = self.get_param("y")
        cov_max = self.get_param("covmax")
        cov_min = self.get_param("covmin")
        low_memory = self.get_param("low_memory")

        rules_list = self.find_candidates(length)

        if len(rules_list) > 0:
            if nb_jobs == 1:
                rs = [
                    eval_rule(
                        rule, x, y, calcmethod, cov_min, cov_max, low_memory
                    )
                    for rule in rules_list
                ]
            else:
                rs = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                    delayed(eval_rule)(
                        rule, x, y, calcmethod, cov_min, cov_max, low_memory
                    )
                    for rule in rules_list
                )

            rs = list(filter(None, rs))
            rs_length_up = RuleSet(rs)
            rs_length_up = rs_length_up.drop_duplicates()
            return rs_length_up
        else:
            return []

    def find_candidates(self, length):
        """
        Returns the intersection of all suitable rules
        for a given length
        """
        rules_list = []
        ruleset = self.get_param("ruleset")
        cov_min = self.get_param("covmin")
        cov_max = self.get_param("covmax")
        k = self.get_param("k")
        nb_jobs = self.get_param("nb_jobs")
        method = self.get_param("method")
        low_memory = self.get_param("low_memory")
        if low_memory:
            x = self.get_param("x")
        else:
            x = None

        rs1, rs2 = ruleset.get_candidates(x, k, length, method, nb_jobs)
        self.set_params(ruleset=ruleset)

        if len(rs2) > 0:
            inter_list = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                delayed(calc_intersection)(
                    rule, rs1, cov_min, cov_max, x, low_memory
                )
                for rule in rs2
            )

            inter_list = functools.reduce(operator.add, inter_list)

            inter_list = list(filter(None, inter_list))  # to drop bad rules
            inter_list = list(set(inter_list))  # to drop duplicates

            rules_list += inter_list

        return rules_list

    def select_rules(self, length):
        """
        Returns a subset of a given ruleset.
        This subset minimizes the empirical contrast on the learning set
        """
        ymean = self.get_param("ymean")
        # ystd = self.get_param('ystd')
        ruleset = self.get_param("ruleset")
        beta = self.get_param("beta")
        epsilon = self.get_param("epsilon")

        x_train = self.get_param("X")

        if length > 0:
            sub_ruleset = ruleset.extract_length(length)
        else:
            sub_ruleset = copy.deepcopy(ruleset)

        print("Number of rules: %s" % str(len(sub_ruleset)))

        if hasattr(self, "sigma"):
            sigma = self.get_param("sigma")
        else:
            sigma = min(sub_ruleset.get_rules_param("var"))
            self.set_params(sigma=sigma)

        significant_list = list(
            filter(
                lambda rule: significant_test(rule, ymean, sigma, beta),
                sub_ruleset,
            )
        )
        [rule.set_params(significant=True) for rule in significant_list]
        significant_ruleset = RuleSet(significant_list)
        print(
            "Number of rules after significant test: %s"
            % str(len(significant_ruleset))
        )

        if len(significant_ruleset) > 0:
            significant_ruleset.sort_by("cov", True)
            # significant_ruleset.sort_by('crit', False)
            rg_add, selected_rs = self.select(significant_ruleset)
            print("Number of selected significant rules: %s" % str(rg_add))

        else:
            selected_rs = None
            print("No significant rules selected!")

        if self.coverage:
            # Add insignificant rules
            if selected_rs is None or selected_rs.calc_coverage(x_train) < 1:
                insignificant_list = filter(
                    lambda rule: insignificant_test(rule, sigma, epsilon),
                    sub_ruleset,
                )
                insignificant_list = list(
                    filter(
                        lambda rule: rule not in significant_list,
                        insignificant_list,
                    )
                )
                if len(list(insignificant_list)) > 0:
                    [
                        rule.set_params(significant=False)
                        for rule in insignificant_list
                    ]
                    insignificant_ruleset = RuleSet(insignificant_list)
                    print(
                        "Number rules after insignificant test: %s"
                        % str(len(insignificant_ruleset))
                    )

                    insignificant_ruleset.sort_by("var", False)
                    rg_add, selected_rs = self.select(
                        insignificant_ruleset, selected_rs
                    )
                    print("Number insignificant rules added: %s" % str(rg_add))
                else:
                    print("No insignificant rule added.")
            else:
                print("Covering is completed. No insignificant rule added.")

            # Add rule to have a covering
            if selected_rs.calc_coverage(x_train) < 1:
                print("Warning: Covering is not completed!")
                print(selected_rs.calc_coverage(x_train))
            # TO REINDENT IF UNCOMMENTED (shoud be in the if)
            # neg_rule, pos_rule = add_no_rule(selected_rs, x_train,
            # y_train) features_names = self.get_param('features_names')
            #
            # if neg_rule is not None:
            #     id_feature = neg_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_names))
            #     neg_rule.conditions.set_params(features_names=rule_features)
            #     neg_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0,
            #     cov_max=1.0)
            #     print('Add negative no-rule  %s.' % str(neg_rule))
            #     selected_rs.append(neg_rule)
            #
            # if pos_rule is not None:
            #     id_feature = pos_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_names))
            #     pos_rule.conditions.set_params(features_names=rule_features)
            #     pos_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0,
            #     cov_max=1.0)
            #     print('Add positive no-rule  %s.' % str(pos_rule))
            #     selected_rs.append(pos_rule)
            else:
                print("Covering is completed.")

        return selected_rs

    def select(self, rs, selected_rs=None):
        # TODO extract important lines for IMORA
        # y_train = self.get_param('y')
        # calcmethod = self.get_param('calcmethod')
        # crit_evo = self.get_param('critlist')
        low_memory = self.get_param("low_memory")
        if low_memory:
            x_train = self.get_param("X")
        else:
            x_train = None

        if selected_rs is None:
            selected_rs = RuleSet(rs[:1])
            gamma = self.get_param("gamma")
            i = 1
            rg_add = 1
        else:
            # gamma = self.get_param('gamma')
            gamma = 1.0
            i = 0
            rg_add = 0
        # old_criterion = calc_ruleset_crit(selected_rs,
        #                                   y_train,
        #                                   x_train,
        #                                   calcmethod)
        # crit_evo.append(old_criterion)
        nb_rules = len(rs)

        activation_rs = selected_rs.calc_activation(x_train)
        if low_memory:
            [
                r.set_params(activation=r.get_activation(x_train))
                for r in selected_rs
            ]
        else:
            pass

        while calc_coverage(activation_rs) < 1 and i < nb_rules:
            rs_copy = copy.deepcopy(selected_rs)
            new_rules = rs[i]
            if low_memory:
                new_rules.set_params(
                    activation=new_rules.get_activation(x_train)
                )
            else:
                pass
            union_tests = [
                new_rules.union_test(
                    rule.get_activation(x_train), gamma, x_train
                )
                for rule in rs_copy
            ]

            if all(union_tests) and new_rules.union_test(
                activation_rs, gamma, x_train
            ):
                new_rs = copy.deepcopy(selected_rs)
                new_rs.append(new_rules)
                # new_criterion = calc_ruleset_crit(new_rs, y_train, x_train,
                #                                   calcmethod)

                selected_rs = copy.deepcopy(new_rs)
                activation_rs = selected_rs.calc_activation(x_train)
                # old_criterion = new_criterion
                rg_add += 1

            # crit_evo.append(old_criterion)
            i += 1

        # self.set_params(critlist=crit_evo)
        return rg_add, selected_rs

    def predict(self, x, check_input=True):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected ruleset on X.

        Parameters
        ----------
        x : {array type or sparse matrix or DataFrame or Series
             of shape = [n_samples, n_features]}
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a spares matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        check_input : bool type

        Returns
        -------
        y : {array type of shape = [n_samples]}
            The predicted values.
        """

        # TODO : if df or series, check index ?
        data_x = x
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            data_x = x.values

        y_train = self.get_param("y")
        x_train = self.get_param("X")

        data_x = self.validate_x_predict(data_x, check_input)
        x_copy = self.discretize(data_x)

        ruleset = self.get_param("selected_rs")

        prediction_vector, bad_cells, no_rules = ruleset.predict(
            y_train, x_train, x_copy
        )

        return np.array(prediction_vector), bad_cells, no_rules

    def score(self, x, y, sample_weight=None):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : {array type or sparse matrix or DataFrame or Series
             of shape = [n_samples, n_features]}
            Test samples.

        y : {array type of shape = [n_samples]}
            True values for y.

        sample_weight : {array type of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.
            or
        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """

        # TODO : if df or series, check index ?
        data_x = x
        data_y = y
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            data_x = x.values
        x_copy = copy.copy(data_x)

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            data_y = y.values
        data_y = copy.copy(y)

        prediction_vector = self.predict(x_copy)
        prediction_vector = np.nan_to_num(prediction_vector)

        nan_val = np.argwhere(np.isnan(data_y))
        if len(nan_val) > 0:
            prediction_vector = np.delete(prediction_vector, nan_val)
            data_y = np.delete(data_y, nan_val)

        if len(set(data_y)) == 2:
            th_val = (min(data_y) + max(data_y)) / 2.0
            prediction_vector = list(
                map(
                    lambda p: min(data_y) if p < th_val else max(data_y),
                    prediction_vector,
                )
            )
            return accuracy_score(data_y, prediction_vector)
        else:
            return r2_score(
                data_y,
                prediction_vector,
                sample_weight=sample_weight,
                multioutput="variance_weighted",
            )

    """------   Data functions   -----"""

    def validate_x_predict(self, x, check_input):
        """
        Validate X whenever one tries to predict, apply, predict_proba
        """
        # TODO : if df or series, check index ?
        data_x = x
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            data_x = x.values

        if hasattr(self, "fitted") is False:
            raise AttributeError(
                "Estimator not fitted, "
                "call 'fit' before exploiting the model."
            )

        if check_input:
            # type: np.ndarray
            data_x = check_array(data_x, dtype=None, force_all_finite=False)

            # noinspection PyUnresolvedReferences
            n_features = data_x.shape[1]
            input_features = self.get_param("features_names")
            if len(input_features) != n_features:
                raise ValueError(
                    "Number of features of the model must "
                    "match the input. Model n_features is %s and "
                    "input n_features is %s " % (input_features, n_features)
                )

        return data_x

    def discretize(self, x):
        """
        Used to have discrete values for each series
        to avoid float

        Parameters
        ----------
        x : Union[array, pd.DataFrame, pd.Series]
            shape=[n_samples, n_features]. Features matrix

        Return
        ------
        col : array
            shape=[n_samples, n_features]. Features matrix with each
            features values discretized in nb_bucket values
        """

        # TODO : if df or series, check index ?
        data_x = x
        if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
            data_x = x.values

        nb_col = data_x.shape[1]
        nb_bucket = self.get_param("nb_bucket")
        bins_dict = self.get_param("bins")
        features_names = self.get_param("features_names")

        x_mat = []
        for i in range(nb_col):
            xcol = data_x[:, i]
            try:
                xcol = np.array(xcol.flat, dtype=np.float)
            except ValueError:
                xcol = np.array(xcol.flat, dtype=np.str)

            var_name = features_names[i]

            if np.issubdtype(xcol.dtype, np.floating):
                if var_name not in bins_dict:
                    if len(set(xcol)) >= nb_bucket:
                        bins = find_bins(xcol, nb_bucket)
                        discretized_column = discretize(xcol, nb_bucket, bins)
                        bins_dict[var_name] = bins
                    else:
                        discretized_column = xcol
                else:
                    bins = bins_dict[var_name]
                    discretized_column = discretize(xcol, nb_bucket, bins)
            else:
                discretized_column = xcol

            x_mat.append(discretized_column)

        return np.array(x_mat).T

    def plot_rules(
        self, var1, var2, length=None, col_pos="red", col_neg="blue"
    ):
        """
        Plot the rectangle activation zone of rules in a 2D plot
        the color is corresponding to the intensity of the prediction

        Parameters
        ----------
        var1 : {string type}
               Name of the first variable

        var2 : {string type}
               Name of the second variable

        length : {int type}, optional
                 Option to plot only the length 1 or length 2 rules

        col_pos : {string type}, optional,
                  Name of the color of the zone of positive rules

        col_neg : {string type}, optional
                  Name of the color of the zone of negative rules

        -------
        Draw the graphic
        """
        selected_rs = self.get_param("selected_rs")
        nb_bucket = self.get_param("nb_bucket")

        if length is not None:
            sub_ruleset = selected_rs.extract_length(length)
        else:
            sub_ruleset = selected_rs

        plt.plot()

        for rule in sub_ruleset:
            rule_condition = rule.conditions

            var = rule_condition.get_param("features_index")
            bmin = rule_condition.get_param("bmin")
            bmax = rule_condition.get_param("bmax")
            length_rule = rule.get_param("length")

            if rule.get_param("pred") > 0:
                hatch = "/"
                facecolor = col_pos
                alpha = min(1, abs(rule.get_param("pred")) / 2.0)
            else:
                hatch = "\\"
                facecolor = col_neg
                alpha = min(1, abs(rule.get_param("pred")) / 2.0)

            if length_rule == 1:
                if var[0] == var1:
                    p = patches.Rectangle(
                        (bmin[0], 0),  # origin
                        (bmax[0] - bmin[0]) + 0.99,  # width
                        nb_bucket,  # height
                        hatch=hatch,
                        facecolor=facecolor,
                        alpha=alpha,
                    )
                    plt.gca().add_patch(p)

                elif var[0] == var2:
                    p = patches.Rectangle(
                        (0, bmin[0]),
                        nb_bucket,
                        (bmax[0] - bmin[0]) + 0.99,
                        hatch=hatch,
                        facecolor=facecolor,
                        alpha=alpha,
                    )
                    plt.gca().add_patch(p)

            elif length_rule == 2:
                if var[0] == var1 and var[1] == var2:
                    p = patches.Rectangle(
                        (bmin[0], bmin[1]),
                        (bmax[0] - bmin[0]) + 0.99,
                        (bmax[1] - bmin[1]) + 0.99,
                        hatch=hatch,
                        facecolor=facecolor,
                        alpha=alpha,
                    )
                    plt.gca().add_patch(p)

                elif var[1] == var1 and var[0] == var2:
                    p = patches.Rectangle(
                        (bmin[1], bmin[0]),
                        (bmax[1] - bmin[1]) + 0.99,
                        (bmax[0] - bmin[0]) + 0.99,
                        hatch=hatch,
                        facecolor=facecolor,
                        alpha=alpha,
                    )
                    plt.gca().add_patch(p)

        if length is None:
            plt.gca().set_title("rules activations")
        else:
            plt.gca().set_title("rules l%s activations" % str(length))

        plt.gca().axis([-0.1, nb_bucket + 0.1, -0.1, nb_bucket + 0.1])

    def plot_pred(
        self,
        x,
        y,
        var1,
        var2,
        cmap=None,
        vmin=None,
        vmax=None,
        add_points=True,
        add_score=False,
    ):
        """
        Plot the prediction zone of rules in a 2D plot

        Parameters
        ----------
        x : {array-like, sparse matrix or DataFrame or Series},
             shape=[n_samples, n_features]
            Features matrix, where n_samples in the number of samples and
            n_features is the number of features.

        y : {array-like or DataFrame or Series}, shape=[n_samples]
            Target vector relative to X

        var1 : {int type}
               Number of the column of the first variable

        var2 : {int type}
               Number of the column of the second variable

        cmap : {colormap object}, optional
               Colormap used for the graphic

        vmax, vmin : {float type}, optional
                     Parameter of the range of the colorbar

        add_points: {boolean type}, optional
                    Option to add the discrete scatter of y

        add_score : {boolean type}, optional
                    Option to add the score on the graphic

        -------
        Draw the graphic
        """

        # TODO : if df or series, check index ?
        data_y = y
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            data_y = y.values
        # No need for checking x type, it is done by self.discretize(x)

        nb_bucket = self.get_param("nb_bucket")
        x_discretized = self.discretize(x)
        selected_rs = self.get_param("selected_rs")
        y_train = self.get_param("y")
        ymean = self.get_param("ymean")
        ystd = self.get_param("ystd")

        x1 = x_discretized[:, var1]
        x2 = x_discretized[:, var2]

        xx, yy = np.meshgrid(range(nb_bucket), range(nb_bucket))

        if cmap is None:
            cmap = plt.cm.get_cmap("coolwarm")

        z = selected_rs.predict(
            y_train,
            np.c_[np.round(xx.ravel()), np.round(yy.ravel())],
            ymean,
            ystd,
        )

        if vmin is None:
            vmin = min(z)
        if vmax is None:
            vmax = max(z)

        z = z.reshape(xx.shape)

        plt.contourf(xx, yy, z, cmap=cmap, alpha=0.8, vmax=vmax, vmin=vmin)

        if add_points:
            area = map(
                lambda b: map(
                    lambda a: np.extract(
                        np.logical_and(x1 == a, x2 == b), data_y
                    ).mean(),
                    range(nb_bucket),
                ),
                range(nb_bucket),
            )
            area = list(area)

            area_len = map(
                lambda b: map(
                    lambda a: len(
                        np.extract(np.logical_and(x1 == a, x2 == b), data_y)
                    )
                    * 10,
                    range(nb_bucket),
                ),
                range(nb_bucket),
            )
            area_len = list(area_len)

            plt.scatter(
                xx,
                yy,
                c=area,
                s=area_len,
                alpha=1.0,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
            )

        plt.title("RIPE prediction")

        if add_score:
            score = self.score(x, data_y)
            plt.text(
                nb_bucket - 0.70,
                0.08,
                ("%.2f" % str(score)).lstrip("0"),
                size=20,
                horizontalalignment="right",
            )

        plt.axis([-0.01, nb_bucket - 0.99, -0.01, nb_bucket - 0.99])
        plt.colorbar()

    def plot_counter_variables(self):
        """
        Function plots a graphical counter of variables used in rules.
        """
        rs = self.get_param("selected_rs")
        f = rs.plot_counter_variables()

        return f

    def plot_counter(self):
        """
        Function plots a graphical counter of variables used in rules by
        modality.
        """
        nb_bucket = self.get_param("nb_bucket")
        y_labels, counter = self.make_count_matrix(return_vars=True)

        x_labels = list(map(lambda i: str(i), range(nb_bucket)))

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(
            counter,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="Reds",
            linewidths=0.05,
            ax=ax,
            center=0.0,
        )
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def plot_dist(self, x=None):
        """
        Function plots a graphical correlation of rules.
        """
        # No need to check x type, it is done in plot_dist()
        rs = self.get_param("selected_rs")
        if x is None and self.get_param("low_memory"):
            x = self.get_param("X")

        f = rs.plot_dist(x=x)

        return f

    def plot_intensity(self):
        """
        Function plots a graphical counter of variables used in rules.
        """
        y_labels, counter = self.make_count_matrix(return_vars=True)
        intensity = self.make_count_matrix(add_pred=True)

        nb_bucket = self.get_param("nb_bucket")
        x_labels = [str(i) for i in range(nb_bucket)]

        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.divide(intensity, counter)

        val[np.isneginf(val)] = np.nan
        val = np.nan_to_num(val)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(
            val,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="bwr",
            linewidths=0.05,
            ax=ax,
            center=0.0,
        )
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def make_count_matrix(self, add_pred=False, return_vars=False):
        """
        Return a count matrix of each variable in each modality
        """
        ruleset = self.get_param("selected_rs")
        nb_bucket = self.get_param("nb_bucket")

        counter = get_variables_count(ruleset)

        vars_list = [item[0] for item in counter]

        count_mat = np.zeros((nb_bucket, len(vars_list)))
        str_id = []

        for rule in ruleset:
            cd = rule.conditions
            var_name = cd.get_param("features_names")
            bmin = cd.get_param("bmin")
            bmax = cd.get_param("bmax")

            for j in range(len(var_name)):
                if type(bmin[j]) != str:
                    for b in range(int(bmin[j]), int(bmax[j]) + 1):
                        var_id = vars_list.index(var_name[j])
                        if add_pred:
                            count_mat[b, var_id] += rule.get_param("pred")
                        else:
                            count_mat[b, var_id] += 1
                else:
                    str_id += [vars_list.index(var_name[j])]

        vars_list = [i for j, i in enumerate(vars_list) if j not in str_id]
        count_mat = np.delete(count_mat.T, str_id, 0)

        if return_vars:
            return vars_list, count_mat
        else:
            return count_mat

    def make_selected_df(self):
        """
        Returns
        -------
        selected_df : {DataFrame type}
                      DataFrame of selected RuleSet for presentation
        """
        selected_rs = self.get_param("selected_rs")
        selected_df = selected_rs.make_selected_df()
        return selected_df

    """------   Getters   -----"""

    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, "Must be a string"
        if hasattr(self, param):
            return getattr(self, param)
        else:
            return None

    """------   Setters   -----"""

    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


"""
---------
Functions
---------
"""


def make_condition(rule):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.

    Parameters
    ----------
    rule : {rule type}
           A rule

    Return
    ------
    conditions_str : {str type}
                     A new string for the condition of the rule
    """
    conditions = rule.get_param("conditions").get_attr()
    length = rule.get_param("length")
    conditions_str = ""
    for i in range(length):
        if i > 0:
            conditions_str += " & "

        conditions_str += conditions[0][i]
        if conditions[2][i] == conditions[3][i]:
            conditions_str += " = "
            conditions_str += str(conditions[2][i])
        else:
            conditions_str += r" $\in$ ["
            conditions_str += str(conditions[2][i])
            conditions_str += ", "
            conditions_str += str(conditions[3][i])
            conditions_str += "]"

    return conditions_str


def make_rules(
    feature_name, feature_index, x, y, method, cov_min, cov_max, low_memory
):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.

    Parameters
    ----------
    feature_name : {string type}
        Name of the feature

    feature_index : {int type}
        Columns index of the feature

    x: Union[array-like, pd.DataFrame, pd.Series]
        shape = [n, d]. The training input samples after discretization.

    y: Union[array-like, pd.DataFrame, pd.Series]
        shape = [n]. The normalized target values (real numbers).

    method: str
        The method mse_function or mse_function criterion

    cov_min: float
        Such as 0 <= covmin <= 1. The minimal coverage of one rule

    cov_max: float
        Such as 0 <= covmax <= 1. The maximal coverage of one rule

    low_memory: bool
        To save activation vectors of rules

    Return
    ------
    rules_list: List
        the list of all suitable rules on the chosen feature.
    """

    # TODO : if df or series, check index ?
    data_x = x
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        data_x = x.values
    xcol = data_x[:, feature_index]
    # No need to check y type since it is done in eval_rules

    try:
        xcol = np.array(xcol, dtype=np.float)
        notnan_vect = np.extract(np.isfinite(xcol), xcol)
        values = list(map(float, np.sort(list(set(notnan_vect)))))
    except ValueError:
        xcol = np.array(xcol, dtype=np.str)
        values = list(map(str, np.sort(list(set(xcol)))))

    rules_list = []
    for bmin in values:
        j = values.index(bmin)
        if xcol.dtype != np.str:
            for bmax in values[j:]:
                conditions = RuleConditions(
                    features_names=[feature_name],
                    features_index=[feature_index],
                    bmin=[bmin],
                    bmax=[bmax],
                    xmax=[max(values)],
                    xmin=[min(values)],
                    values=values,
                )

                rule = Rule(conditions)
                rules_list.append(
                    eval_rule(
                        rule, data_x, y, method, cov_min, cov_max, low_memory
                    )
                )

        else:
            bmax = bmin
            conditions = RuleConditions(
                features_names=[feature_name],
                features_index=[feature_index],
                bmin=[bmin],
                bmax=[bmax],
                xmax=[max(values)],
                xmin=[min(values)],
                values=values,
            )

            rule = Rule(conditions)
            rules_list.append(
                eval_rule(
                    rule, data_x, y, method, cov_min, cov_max, low_memory
                )
            )

    rules_list = list(filter(None, rules_list))
    return rules_list


def eval_rule(
    rule: Rule,
    x: Union[np.ndarray, pd.DataFrame, pd.Series],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    method: str,
    cov_min: float,
    cov_max: float,
    low_memory: bool,
):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : RICE.Rule
        An rule object (it means with condition on X)

    x : Union[np.ndarray, pd.DataFrame, pd.Series]
        The training input samples after discretization.
        shape = [n, d]. So date-features

    y : Union[np.ndarray, pd.DataFrame, pd.Series]
        The normalized target values (real numbers).
        shape = [n]

    method : str
        The methode mse_function or mse_function criterion

    cov_min : float
        Must be such as 0 <= covmin <= 1
        The minimal coverage of one rule

    cov_max : float
        Must be such as 0 <= covmin <= 1
        The maximal coverage of one rule

    x : Union[np.ndarray, None]
        The training input samples after discretization.
        Shape = [n, d].
        If low_memory is True x must not be None

    low_memory : bool
        To save activation vectors of rules

    Return
    ------
    None
        if the rule does not verified criteria

    rule : Rule
        rule with all statistics calculated

    """

    # No need to check X and y type, it is done in calc_stats

    rule.calc_stats(
        x=x,
        y=y,
        method=method,
        cov_min=cov_min,
        cov_max=cov_max,
        low_memory=low_memory,
    )

    if rule.get_param("out") is False:
        return rule
    else:
        return None


def calc_intersection(
    rule, ruleset, cov_min, cov_max, x=None, low_memory=False
):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : {rule type}
             An rule object

    ruleset : {ruleset type}
                 A set of rule

    cov_min : {float type such as 0 <= covmin <= 1}
              The maximal coverage of one rule

    cov_max : {float type such as 0 <= covmax <= 1}
              The maximal coverage of one rule

    x : {array-like or discretized matrix or DataFrame or Series,
        shape = [n, d] or None}
        The training input samples after discretization.
        If low_memory is True X must not be None

    low_memory : {bool type}
                 To save activation vectors of rules

    Return
    ------
    rules_list : {list type}
                 List of rule made by intersection of rule with
                 rules from the rules set ruleset_l1.

    """

    # No need to check X and y type, it is done in intersect

    rules_list = [
        rule.intersect(r, cov_min, cov_max, x, low_memory) for r in ruleset
    ]
    rules_list = list(filter(None, rules_list))  # to drop bad rules
    rules_list = list(set(rules_list))
    return rules_list


def calc_ruleset_crit(ruleset, y_train, x_train=None, method="MSE"):
    """
    Calculation of the criterion of a set of rule

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    y_train : {array-like or DataFrame or Series, shape = [n]}
           The normalized target values (real numbers).

    x_train : {array-like or DataFrame or Series, shape = [n]}
              The normalized target values (real numbers).

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           The value of the criteria for the method
    """

    # No need to check X and y type, it is done in calc_pred and calc_criterion

    prediction_vector, bad_cells, no_rules = ruleset.calc_pred(
        y_train=y_train, x_train=x_train
    )
    criterion = calc_criterion(prediction_vector, y_train, method)
    return criterion


def find_cluster(ruleset, x, k, n_jobs):

    # No need to check X and y type, it is done in get_activation

    if len(ruleset) > k:
        prediction_matrix = np.array(
            [
                rule.get_param("pred") * rule.get_activation(x)
                for rule in ruleset
            ]
        )

        cluster_algo = KMeans(n_clusters=k, n_jobs=n_jobs)
        cluster_algo.fit(prediction_matrix)
        return cluster_algo.labels_
    else:
        return range(len(ruleset))


def select_candidates(ruleset, k):
    """
    Returns a set of candidates to increase length
    with a maximal number k
    """
    rules_list = []
    for i in range(k):
        sub_rs = ruleset.extract("cluster", i)
        if len(sub_rs) > 0:
            sub_rs.sort_by("var", True)
            rules_list.append(sub_rs[0])

    return RuleSet(rules_list)


def get_variables_count(ruleset):
    """
    Get a counter of all different features in the ruleset

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    Return
    ------
    count : {Counter type}
            Counter of all different features in the ruleset
    """
    col_varuleset = [
        rule.conditions.get_param("features_names") for rule in ruleset
    ]
    varuleset_list = functools.reduce(operator.add, col_varuleset)
    count = Counter(varuleset_list)

    count = count.most_common()
    return count


def dist(u, v):
    """
    Compute the distance between two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A predictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v), "The two array must have the same length"
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u), np.dot(v, v))
    return 1 - num / deno


def mse_function(prediction_vector, y):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type or DataFrame or Series}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """

    data_y = y
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        data_y = y.values

    assert len(prediction_vector) == len(
        data_y
    ), "The two array must have the same length"
    error_vector = prediction_vector - data_y
    criterion = np.nanmean(error_vector ** 2)
    return criterion


def mae_function(prediction_vector, y):
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type or DataFrame or Series}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean absolute error
    """

    data_y = y
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        data_y = y.values

    assert len(prediction_vector) == len(
        data_y
    ), "The two array must have the same length"
    error_vect = np.abs(prediction_vector - data_y)
    criterion = np.nanmean(error_vect)
    return criterion


def aae_function(prediction_vector, y):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type or DataFrame or Series}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """

    data_y = y
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        data_y = y.values

    assert len(prediction_vector) == len(
        data_y
    ), "The two array must have the same length"
    error_vector = np.mean(np.abs(prediction_vector - data_y))
    median_error = np.mean(np.abs(data_y - np.median(data_y)))
    return error_vector / median_error


def calc_criterion(prediction_vector, y, method="mse"):
    """
    Compute the criteria

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type or DataFrame or Series}
        The real target values (real numbers)

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           Criteria value
    """

    data_y = y
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        data_y = y.values

    y_fillna = np.nan_to_num(data_y)

    if method == "mse":
        criterion = mse_function(prediction_vector, y_fillna)

    elif method == "mae":
        criterion = mae_function(prediction_vector, y_fillna)

    elif method == "aae":
        criterion = aae_function(prediction_vector, y_fillna)

    else:
        raise "Method %s unknown" % method

    return criterion


def significant_test(rule, ymean, sigma, beta):
    """
    Parameters
    ----------
    rule : {Rule type}
        A rule.

    ymean : {float type}
            The mean of y.

    sigma : {float type}
            The noise estimator.

    beta : {float type}
            The beta factor.

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    left_term = beta * abs(rule.get_param("pred") - ymean)
    right_term = np.sqrt(max(0, rule.get_param("var") - sigma))
    return left_term > right_term


def insignificant_test(rule, sigma, epsilon):
    return epsilon >= np.sqrt(max(0, rule.get_param("var") - sigma))


def calc_coverage(vect: Union[np.array, pd.Series]):
    """
    Compute the coverage rate of an activation vector

    Parameters
    ----------
    vect : Union[np.array, pd.Series]
        An activation vector. It means a sparse array with two
        different values 0, if the rule is not active
        and the 1 is the rule is active.

    Return
    ------
    cov : {float type}
          The coverage rate
    """
    return np.dot(vect, vect) / float(vect.size)


def get_activated_values(
    activation_vector: Union[np.array, pd.Series],
    y: Union[np.array, pd.Series],
):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    activation_vector : Union[np.array, pd.Series]
        An activation vector. It means a sparse array with two
        different values 0, if the rule is not active
        and the 1 is the rule is active.
        Shape=[n, m]. So date-Y names

    y : Union[np.array, pd.Series]
        The target values (real numbers), shape=[n, m]. So date-Y names.

    Return
    ------
    np.array
        the list of Y values that are activated
    """
    if type(activation_vector) != type(y):
        raise ValueError(
            "Activation vector and Y must be of same type."
            f"\nHere activation vector is a {type(activation_vector)} while "
            f"Y is a {type(y)} "
        )

    if isinstance(y, pd.Series):
        return y[activation_vector != 0]
    else:
        return np.extract(activation_vector != 0, y)


def calc_prediction(
    activation_vector: Union[np.array, pd.Series],
    y: Union[np.array, pd.Series],
):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    activation_vector : Union[np.array, pd.Series]
        An activation vector. It means a sparse array with two
        different values 0, if the rule is not active
        and the 1 is the rule is active.
        Shape=[n, m]. So date-Y names

    y : Union[np.array, pd.Series]
        The target values (real numbers), shape=[n, m]. So date-Y names.

    Return
    ------
    predictions : float
        The empirical conditional expectation of y knowing X
    """

    y_cond = get_activated_values(activation_vector, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        return np.nanmean(y_cond)


def calc_variance(activation_vector, y):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    activation_vector : Union[np.array, pd.Series]
        An activation vector. It means a sparse array with two
        different values 0, if the rule is not active
        and the 1 is the rule is active.
        Shape=[n, m]. So date-Y names

    y : Union[np.array, pd.Series]
        The target values (real numbers), shape=[n, m]. So date-Y names.

    Return
    ------
    predictions : float
        The empirical conditional variance of y knowing X
    """

    y_cond = get_activated_values(activation_vector, y)
    cond_var = np.var(y_cond)
    return cond_var


# noinspection PyArgumentList
def find_bins(x, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    x : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                Number of modalities

    Return
    ------
    bins : {ndarray type}
           The bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
    bins = np.array([np.nanpercentile(x, i) for i in q_list])

    if bins.min() != 0:
        test_bins = bins / bins.min()
    else:
        test_bins = bins

    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(x, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins / bins.min()
        else:
            test_bins = bins

    return bins


def discretize(x, nb_bucket, bins=None):
    """
    Function used to have discretize xcol in nb_bucket values
    if xcol is a real series and do nothing if xcol is a string series

    Parameters
    ----------
    x : {Series type}
           Series to discretize

    nb_bucket : {int type}
                Number of modalities

    bins : {ndarray type}, optional, default None
           If you have already calculate the bins for xcol

    Return
    ------
    x_discretized : {Series type}
                       The discretization of xcol
    """
    if np.issubdtype(x.dtype, np.floating):
        # extraction of the list of xcol values
        notnan_vector = np.extract(np.isfinite(x), x)
        nan_index = ~np.isfinite(x)
        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vector)) >= nb_bucket or bins is not None:
            if bins is None:
                bins = find_bins(x, nb_bucket)
            # discretization of the xcol with bins
            # x_discretized = np.digitize(x, bins=bins)
            # x_discretized = np.array(x_discretized, dtype='float')
            # Quicker than np.digitize
            x_discretized = bisect(bins, x)

            if sum(nan_index) > 0:
                x_discretized[nan_index] = np.nan

            return x_discretized

        return x

    else:
        return x
