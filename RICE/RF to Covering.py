import numpy as np
import tqdm
import copy
import math
from operator import itemgetter

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree

import RICE


def extract_rules_from_tree(tree, features, xmin, xmax):
    dt = tree.tree_
    
    def visitor(node, depth, cond=None, rule_list=None):
        if rule_list is None:
            rule_list = []
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            # If
            new_cond = RICE.RuleConditions([features[dt.feature[node]]],
                                           [dt.feature[node]],
                                           bmin=[xmin[dt.feature[node]]],
                                           bmax=[dt.threshold[node]],
                                           xmin=[xmin[dt.feature[node]]],
                                           xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    
                    new_cond = RICE.RuleConditions(features_name=conditions_list[0],
                                                   features_index=conditions_list[1],
                                                   bmin=conditions_list[2],
                                                   bmax=conditions_list[3],
                                                   xmax=conditions_list[5],
                                                   xmin=conditions_list[4])
                else:
                    new_bmax = dt.threshold[node]
                    new_cond = copy.deepcopy(cond)
                    place = cond.features_index.index(dt.feature[node])
                    new_cond.bmax[place] = min(new_bmax, new_cond.bmax[place])
                    
            # print (RICE.Rule(new_cond))
            new_rg = RICE.Rule(copy.deepcopy(new_cond))
            rule_list.append(new_rg)                    
            
            rule_list = visitor(dt.children_left[node], depth + 1,
                                new_cond, rule_list)
            
            # Else
            new_cond = RICE.RuleConditions([features[dt.feature[node]]],
                                           [dt.feature[node]],
                                           bmin=[dt.threshold[node]],
                                           bmax=[xmax[dt.feature[node]]],
                                           xmin=[xmin[dt.feature[node]]],
                                           xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    new_cond = RICE.RuleConditions(features_name=conditions_list[0],
                                                   features_index=conditions_list[1],
                                                   bmin=conditions_list[2],
                                                   bmax=conditions_list[3],
                                                   xmax=conditions_list[5],
                                                   xmin=conditions_list[4])
                else:
                    new_bmin = dt.threshold[node]
                    new_bmax = xmax[dt.feature[node]]
                    new_cond = copy.deepcopy(cond)
                    place = new_cond.features_index.index(dt.feature[node])
                    new_cond.bmin[place] = max(new_bmin, new_cond.bmin[place])
                    new_cond.bmax[place] = max(new_bmax, new_cond.bmax[place])
                    
            # print (RICE.Rule(new_cond))
            new_rg = RICE.Rule(copy.deepcopy(new_cond))
            rule_list.append(new_rg)
            
            rule_list = visitor(dt.children_right[node], depth + 1, new_cond, rule_list)

        return rule_list
    
    rule_list = visitor(0, 1)
    return rule_list


def select_rs(rs, gamma=1.0, selected_rs=None):
    """
    Returns a subset of a given rs. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    # Then optimization
    if selected_rs is None or len(selected_rs) == 0:
        selected_rs = RICE.RuleSet(rs[:1])
        id_rule = 1
    else:
        id_rule = 0
    
    nb_rules = len(rs)
    
    for i in tqdm.tqdm(range(id_rule, nb_rules), desc='Selection'):
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rs[i]
        
        utest = [new_rules.union_test(rule.get_activation(),
                                      gamma)
                 for rule in rs_copy]
        
        if all(utest) and new_rules.union_test(selected_rs.calc_activation(),
                                               gamma):
            new_rs = copy.deepcopy(selected_rs)
            new_rs.append(new_rules)
            
            selected_rs = copy.deepcopy(new_rs)
    
    return selected_rs


def get_norule(rs, X, y):
    """
    Return the two smallest rule of CP1 that cover all none covered
    positive and negative points
    
    Parameters
    ----------
    rs : {RuleSet type}
         A set of rules
         
    X : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
    
        The normalized target values (real numbers).
        
    Return
    ------
    neg_rule, pos_rule : {tuple type}
                         Two rules or None
    """
    no_rule_act = 1 - rs.calc_activation()
    norule = None
    if sum(no_rule_act) > 0:
        norule_list = get_norules_list(no_rule_act, X, y)

        if len(norule_list) > 0:
            norule = norule_list[0]
            for rg in norule_list[1:]:
                conditions_list = norule.intersect_conditions(rg)
                new_conditions = RICE.RuleConditions(features_name=conditions_list[0],
                                                     features_index=conditions_list[1],
                                                     bmin=conditions_list[2],
                                                     bmax=conditions_list[3],
                                                     xmax=conditions_list[5],
                                                     xmin=conditions_list[4])
                norule = RICE.Rule(new_conditions)

    return norule
    

def get_norules_list(no_rule_act, X, y):
    norule_list = []
    for i in range(X.shape[1]):
        try:
            sub_x = X[:, i].astype('float')
        except ValueError:
            sub_x = None
        
        if sub_x is not None:
            sub_no_rule_act = no_rule_act[~np.isnan(sub_x)]
            sub_x = sub_x[~np.isnan(sub_x)][sub_no_rule_act]
            sub_x = np.extract(sub_no_rule_act, sub_x)
            
            norule = RICE.Rule(RICE.RuleConditions(bmin=[sub_x.min()],
                                                   bmax=[sub_x.max()],
                                                   features_name=[''],
                                                   features_index=[i],
                                                   xmax=[sub_x.max()],
                                                   xmin=[sub_x.min()]))
            norule_list.append(norule)
                
    return norule_list


def get_significant(rules_list, ymean, beta, gamma, sigma2):
    significant_rules = list(filter(lambda rule: beta * abs(ymean - rule.pred)
                                                    > math.sqrt(max(0, rule.var - sigma2)),
                                    rules_list))

    print('Nb of significant rules', len(significant_rules))
    significant_rs = RICE.RuleSet(significant_rules)
    print('Coverage rate of significant rule:', significant_rs.calc_coverage())

    significant_rs.sort_by(crit='cov', maximized=True)
    if len(significant_rs) > 0:
        significant_selected_rs = select_rs(rs=significant_rs, gamma=gamma)
    else:
        significant_selected_rs = RICE.RuleSet([])

    print('Nb of selected rules ', len(significant_selected_rs))
    print('Coverage rate of the selected RuleSet ', significant_selected_rs.calc_coverage())
    
    return significant_selected_rs


def add_insignificant_rules(rules_list, rs, epsilon, sigma2, gamma):
    insignificant_rule = list(filter(lambda rule: epsilon > math.sqrt(max(0, rule.var - sigma2)),
                                     rules_list))
    print('Nb of insignificant rules', len(insignificant_rule))
    insignificant_rs = RICE.RuleSet(insignificant_rule)
    print('Coverage rate of significant rule:', insignificant_rs.calc_coverage())
    
    if len(insignificant_rs) > 0:
        insignificant_rs.sort_by(crit='var', maximized=False)
        selected_rs = select_rs(rs=insignificant_rs, gamma=gamma,
                                selected_rs=rs)
    else:
        selected_rs = RICE.RuleSet([])
        
    print('Number of rules :', len(selected_rs))
    print('Coverage rate of the selected RuleSet ', selected_rs.calc_coverage())
    
    return selected_rs


def add_norule(rs, y, X, features):
    new_rs = copy.deepcopy(rs)
    if rs.calc_coverage() < 1.0:
        no_rule = get_norule(copy.deepcopy(rs), X, y)
        
        if no_rule is not None:
            id_feature = no_rule.conditions.get_param('features_index')
            rule_features = list(itemgetter(*id_feature)(features))
            no_rule.conditions.set_params(features_name=rule_features)
            no_rule.calc_stats(y=y, x=X, cov_min=0.0, cov_max=1.1)
            new_rs.append(no_rule)
    
    return new_rs


def find_covering(rules_list, X, y, sigma2=None,
                  alpha=1./2 - 1/100,
                  gamma=0.95):

    n_train = len(y)
    cov_min = n_train ** (-alpha)
    print('Minimal coverage rate:', cov_min)

    sub_rules_list = list(filter(lambda rule: rule.cov >= cov_min, rules_list))
    print('Nb of rules with good coverage rate:', len(sub_rules_list))

    if sigma2 is None:
        var_list = [rg.var for rg in sub_rules_list]
        sigma2 = min(list(filter(lambda v: v > 0, var_list)))
        print('Sigma 2 estimation', sigma2)
        
    beta = pow(n_train, alpha / 2. - 1. / 4)
    print('Beta coefficient:', beta)
    epsilon = beta * np.std(y)
    print('Epsilon coefficient:', epsilon)
    
    significant_selected_rs = get_significant(sub_rules_list, np.mean(y), beta, gamma, sigma2)
    
    if significant_selected_rs.calc_coverage() < 1.0:
        selected_rs = add_insignificant_rules(sub_rules_list, significant_selected_rs,
                                              epsilon, sigma2, gamma)
        
        if selected_rs.calc_coverage() < 1.0:
            new_rs = add_norule(selected_rs, y, X, features)
        else:
            print('No norule added')
            new_rs = copy.copy(selected_rs)
    else:
        print('Significant rules form a covering')
        selected_rs = copy.copy(significant_selected_rs)
        new_rs = copy.copy(significant_selected_rs)
        
    return significant_selected_rs, selected_rs, new_rs


if __name__ == '__main__':
    # Data parameters
    seed = 42
    np.random.seed(seed)
    sigma = None
    test_size = 0.3
    
    # RF parameters
    nb_tree = 50
    K = 2000
    number_nodes = int(K/2. + 1)
    max_depth = int(np.floor(np.log(number_nodes) / np.log(2)))
    
    # Covering parameters
    alpha = 1./2 - 1/100.
    gamma = 0.95
    
    import pandas as pd
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    
    y = boston['MEDV']
    X = boston.drop(['MEDV'], axis=1)
    
    features = X.columns
    X = X.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    if test_size == 0.0:
        X_test = X_train
        y_test = y_train

    regr_rf = RandomForestRegressor(n_estimators=nb_tree, random_state=seed, max_depth=max_depth)
    regr_rf.fit(X_train, y_train)
    
    rule_list = []
    for tree in tqdm.tqdm(regr_rf.estimators_, desc='Rules extraction'):
        rule_list += extract_rules_from_tree(tree, features,
                                             X_train.min(axis=0),
                                             X_train.max(axis=0))
    
    print('Nb of rules generated by the Random Forstes:', len(rule_list))
    
    temp = [rule.calc_stats(y=y_train, x=X_train, cov_min=0.0, cov_max=1.0)
            for rule in tqdm.tqdm(rule_list, desc='Rules evaluation')]
    del temp
    
    significant_selected_rs, selected_rs, new_rs = find_covering(rule_list, X_train, y_train, sigma, alpha, gamma)

    pred_rf = regr_rf.predict(X_test)
    pred_significant = significant_selected_rs.predict(y_train, X_train, X_test)
    pred = selected_rs.predict(y_train, X_train, X_test)
    pred_norule = new_rs.predict(y_train, X_train, X_test)

    print('Radom Forest R2 score', r2_score(y_test, pred_rf))
    print('Significant rules R2 score', r2_score(y_test, pred_significant))
    print('Significant and insignificant rules R2 score', r2_score(y_test, pred))
    print('Covering R2 score', r2_score(y_test, pred_norule))
    print()
    print('Radom Forest MSE', np.mean(pow(y_test - pred_rf, 2)))
    print('Significant rules MSE', np.mean(pow(y_test - pred_significant, 2)))
    print('Significant and insignificant rules MSE', np.mean(pow(y_test - pred, 2)))
    print('Covering MSE', np.mean(pow(y_test - pred_norule, 2)))
