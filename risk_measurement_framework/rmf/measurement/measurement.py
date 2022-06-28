from keras import backend as K
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams.update({'font.size': 10})
from itertools import cycle
from measurement.log import *

risk_indicators_raw = ("Accuracy",
                       "cpu",
                       "ram",
                       "gpu")
risk_indicators = ("attackers_goal",
                   "attackers_knowledge",
                   "attack_time",
                   "found_pattern",
                   "counter",
                   "poisoned_images",
                   "tp",
                   "tn",
                   "fp",
                   "fn")

def separating_measures(low_l, high_l) -> list:

    if low_l is None:
        raise IndexError('Low-level attributes cannot be empty!')
    elif high_l is None:
        raise IndexError('High-Level attributes cannot be empty!')

    base_mea_raw = {}
    base_measures = {}

    for i, j in low_l.items():
        for item_raw in risk_indicators_raw:
            if j is item_raw:
                base_mea_raw[i] = j
            else:
                base_measures[i] = j
    log(f"Low-level base measures {base_measures}, Low-level base measures raw {base_mea_raw}")

    for k, l in high_l.items():
        for item in risk_indicators:
            if l is item:
                base_measures[k] = l
            else:
                base_mea_raw[k] = l
    log(f"High-level base measures {base_measures}, High-level base measures raw {base_mea_raw}")

    return base_mea_raw, base_measures

def measurement_functions(base_measures, y_test, y_score, n_classes, cm) -> list:

    def __ml_metrics(y_test, y_score):

        precision = dict()
        recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision vs. recall curve")
        plt.figure(figsize = (20,20))
        plt.show()

        df_cm = pd.DataFrame(cm, index = classes,  columns = classes)
        plt.figure(figsize = (20,20))
        sns.heatmap(df_cm, annot=True)

        plt.show()

        f1 = f1_score(y_test, y_score, average=None)

        return precision, recall, f1

    def __attack_steps(base_measures):

        steps = 0

        damage_indicators = ("attackers_goal",
                             "attackers_knowledge",
                             "counter",
                             "attack_time")

        return steps

    def __extent_of_damage(base_measures):

        dmg = 0.0

        damage_indicators = ("found_pattern",
                             "poisoned_images",
                             "tp",
                             "tn",
                             "fp",
                             "fn")


        return dmg

    precision, recall, f1 = __ml_metrics(y_test, y_score)
    attack_steps = __attack_steps(base_measures)
    dmg = __extent_of_damage(base_measures)

    derived_measures = [precision, recall, f1, attack_steps, dmg]
    return derived_measures

def analytical_model(base_mea_raw: list(), derived_measures: list()):

    list_base_measure = list()
    list_derived_measures = list()

    def __calc_eff():

        list_base_measure = [list_base_measure.append(based_measure)
                             for base_measure in base_mea_raw
                             if base_measure is "yes"]
        list_derived_measures = [list_derived_measures.append(derived_measure)
                                 for derived_measure in derived_measures
                                 if derived_measure is "no"]

        sum_base = sum(list_base_measure)
        sum_derived = sum(list_derived_measures)

        attackers_effort = sum_base + sum_derived

        return attackers_effort

    def __calc_extent():
        list_base_measure = [list_base_measure.append(based_measure)
                             for base_measure in base_mea_raw
                             if base_measure is "yes"]
        list_derived_measures = [list_derived_measures.append(derived_measure)
                                 for derived_measure in derived_measures
                                 if derived_measure is "no"]

        sum_base = sum(list_base_measure)
        sum_derived = sum(list_derived_measures)

        extent_of_damage = sum_base + sum_derived

        return extent_of_damage

    attackers_effort = __calc_eff()
    extent = __calc_extent()

    return effort, extent

def decision_criteria(*indicator, interval_ext, interval_eff) -> float:

    if indicator > 0 and indicator < interval_ext:
        return indicator[0]
    else:
        raise ValueError(f'Indicator must be between 0 and {intervall_ext}')

    if indicator > 0 and indicator < intervall_eff:
        return indicator[1]
    else:
        raise ValueError(f'Indicator must be between 0 and {intervall_eff}')
