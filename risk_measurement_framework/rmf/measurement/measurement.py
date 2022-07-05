from keras import backend as K
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
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

def measurement_functions(base_measures, y_true, y_score, n_classes, cm, classes, tp, tn, fp, fn, diff, target_label) -> list:

    def __ml_metrics(y_true, y_score, tp, tn, fp, fn):

        precision = dict()
        recall = dict()
        f1_score = list()
        avg_recall = list()

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                                y_score[:, i])

            prec = tp[i] / (tp[i] + fp[i])
            if np.isnan(prec):
                prec = 0
            elif tp[i] == 0 and fp[i] == 0:
                prec = 1

            rec = tp[i] / (tp[i] + fn[i])
            if np.isnan(rec):
                rec = 0
            elif tp[i] == 0 and fn[i] == 0:
                rec = 1

            avg = 2 * prec * rec / (prec + rec)
            avg_recall.append(rec)
            f1_score.append(avg)

            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))


        apr = average_precision_score(y_true, y_score)
        f1 = sum(f1_score)/len(f1_score)
        avg_rec = sum(avg_recall)/len(avg_recall)

        log(f"F1-Score array: {f1}, Average precision score: {apr}")

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision vs. recall curve")
        plt.figure(figsize=(20, 20))

        #plt.show()

        df_cm = pd.DataFrame(cm, index = classes,  columns = classes)
        sns.heatmap(df_cm, annot=True)
        plt.figure(figsize=(20, 20))
        plt.savefig('cm.png')

        ml_metrics = {apr: "apr", avg_rec: "average recall", f1: "f1"}

        return ml_metrics

    def __attack_steps(base_measures):

        steps = 0

        damage_indicators = ("attackers_goal",
                             "attackers_knowledge",
                             "counter",
                             "attack_time")

        for key, value in base_measures.items():
            for indicator in damage_indicators:
                if value is indicator:
                    steps += key

        log(f"Derived measure (attack steps): {steps}")
        return steps

    def __extent_of_damage(base_measures, diff, target_label):

        ml_metrics = __ml_metrics(y_true, y_score, tp, tn, fp, fn)

        dmg = 0.0
        counter = 0
        possible_poisoned = 0
        actual_poisoned = 0.00

        for key, value in base_measures.items():
            if value is "poisoned_images":
                possible_poisoned = key
            elif value is "found_pattern":
                if key == 1:
                    for clean, poison in diff.items():
                        if poison == target_label and clean != target_label:
                            counter += 1
                    actual_poisoned = counter / possible_poisoned


        dmg += sum(list(ml_metrics.keys()))
        dmg += actual_poisoned

        log(f"Derived measure (damage): {dmg}")
        return dmg

    attack_steps = __attack_steps(base_measures)
    dmg = __extent_of_damage(base_measures, diff, target_label)

    derived_measures = [attack_steps, dmg]
    return derived_measures

def analytical_model(base_mea_raw: list(), derived_measures: list()):

    list_base_measure = list()
    list_derived_measures = list()

    def __calc_eff(base_mea_raw, derived_measures):

        for base_measure in base_mea_raw:
            list_base_measure.append(based_measure)

        sum_base = sum(list_base_measure)

        attackers_effort = sum_base + derived_measure[0]

        return attackers_effort

    def __calc_extent(base_mea_raw, derived_measures):

        rev_acc = 0.00

        for key, value in base_mea_raw:
            if value is "Accuracy":
                rev_acc = 10 - value
                log(f"Reversed accuracy: {rev_acc}")

        list_base_measure = [list_base_measure.append(based_measure)
                             for base_measure in base_mea_raw
                             if base_measure is "yes"]
        list_derived_measures = [list_derived_measures.append(derived_measure)
                                 for derived_measure in derived_measures
                                 if derived_measure is "no"]

        sum_base = sum(list_base_measure)
        sum_derived = sum(list_derived_measures)

        extent_of_damage = sum_base + derived_measures[1]

        return extent_of_damage

    attackers_effort = __calc_eff(base_mea_raw, derived_measures)
    extent = __calc_extent(base_mea_raw, derived_measures)

    return effort, extent

def decision_criteria(*indicator, interval_ext, interval_eff) -> float:

    if indicator[0] > 0 and indicator[0] < interval_ext:
        return indicator[0]
    else:
        raise ValueError(f'Indicator must be between 0 and {intervall_ext}')

    if indicator[1] > 0 and indicator[1] < intervall_eff:
        return indicator[1]
    else:
        raise ValueError(f'Indicator must be between 0 and {intervall_eff}')
