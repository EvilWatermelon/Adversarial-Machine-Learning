from keras import backend as K

from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
plt.rcParams.update({'font.size': 10})
from itertools import cycle

from measurement.log import *
from measurement.constants import *

def separating_measures(low_l, high_l) -> list:

    if low_l is None:
        raise IndexError('Low-level attributes cannot be empty!')
    elif high_l is None:
        raise IndexError('High-Level attributes cannot be empty!')

    base_mea_raw = {}
    base_measures = {}

    for i, j in low_l.items():
        for item_raw in RISK_INDICATORS_RAW:
            if j is item_raw:
                base_mea_raw[i] = j
            else:
                base_measures[i] = j
    log(f"Low-level base measures {base_measures}, Low-level base measures raw {base_mea_raw}")

    for k, l in high_l.items():
        for item in RISK_INDICATORS:
            if l is item:
                base_measures[k] = l
            else:
                base_mea_raw[k] = l
    log(f"High-level base measures {base_measures}, High-level base measures raw {base_mea_raw}")

    return base_mea_raw, base_measures

def measurement_functions(base_measures, y_true, y_score, cm, classes, tp, tn, fp, fn, target_label, clean_preds, poison_preds) -> list:

    def __ml_metrics(y_true, y_score, tp, tn, fp, fn):

        precision = dict()
        recall = dict()
        f1_score = list()
        avg_recall = list()
        avg_precision = list()
        plots = list()

        for i in range(len(classes)):
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
            if np.isnan(avg):
                avg = 0
            avg_recall.append(rec)
            avg_precision.append(prec)
            f1_score.append(avg)

            plots.append(plt.plot(recall[i], precision[i], lw=2, label='label {}'.format(i)))

        apr = sum(avg_precision)/len(avg_precision)
        f1 = sum(f1_score)/len(f1_score)
        avg_rec = sum(avg_recall)/len(avg_recall)

        f1 = 10 - f1
        apr = 10 - apr
        avg_rec = 10 - avg_rec

        log(f"F1-Score array: {f1}, Average precision score: {apr}, Average recall score: {avg_rec}")

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(plots[10], ['Label 10'], loc="best")
        plt.title("precision vs. recall curve")

        #plt.show()

        df_cm = pd.DataFrame(cm, index = classes,  columns = classes)
        plt.figure(figsize=(20, 20))
        sns.heatmap(df_cm, annot=True)
        #plt.show()
        #plt.savefig('cm.png')

        ml_metrics = {apr: "apr", avg_rec: "average recall", f1: "f1"}

        return ml_metrics

    def __attack_steps(base_measures):

        steps = 0
        time = 0.0

        damage_indicators = ("attackers_goal",
                             "attackers_knowledge",
                             "counter")

        for key, value in base_measures.items():
            if value is "attack_time":
                time = key
            for indicator in damage_indicators:
                if value is indicator:
                    steps += key

        log(f"Derived measure (attack steps): {steps}")
        return steps, time

    def __extent_of_damage(base_measures, clean_preds, poison_preds, target_label):

        ml_metrics = __ml_metrics(y_true, y_score, tp, tn, fp, fn)

        dmg = 0.0
        counter = 0
        possible_poisoned = 0
        actual_poisoned = 0.00

        for key, value in base_measures.items():
            if value is "poisoned_images":
                possible_poisoned = key

        log(f"Possible images to be poisoned: {possible_poisoned}")

        for i in range(len(y_score)):
            if clean_preds[i] != poison_preds[i] and poison_preds[i] == target_label:
                counter += 1

        log(f"Actual number of Poisoned imaged: {counter}")
        actual_poisoned = counter / possible_poisoned

        log(f"Actual poisoned: {actual_poisoned * 10}")
        dmg += sum(list(ml_metrics.keys()))
        dmg += actual_poisoned * 10

        log(f"Derived measure (damage): {dmg}")
        return dmg

    attack_steps, time = __attack_steps(base_measures)
    dmg = __extent_of_damage(base_measures, clean_preds, poison_preds, target_label)

    derived_measures = (attack_steps, time, dmg)
    return derived_measures

def analytical_model(base_mea_raw, derived_measures):

    raw_names = ("cpu",
                 "ram",
                 "gpu")

    def __calc_eff(base_mea_raw, derived_measures):

        effort_values = list()

        effort_values.append(derived_measures[0])
        effort_values.append(derived_measures[1])

        for key, value in base_mea_raw.items():
            for item in raw_names:
                if value is item:
                    effort_values.append(key)

        return effort_values

    def __calc_extent(base_mea_raw, derived_measures):

        rev_acc = 0.00

        for key, value in base_mea_raw.items():
            if value is "Accuracy":
                rev_acc = 10 - float(key)
                log(f"Reversed accuracy: {rev_acc}")

        extent_of_damage = rev_acc + derived_measures[2]
        log(f"Extent of damage: {extent_of_damage}")

        return extent_of_damage

    attackers_effort = __calc_eff(base_mea_raw, derived_measures)
    extent = __calc_extent(base_mea_raw, derived_measures)

    return attackers_effort, extent

def decision_criteria(*indicator) -> float:

    # Comparing the original with attack values



    measurement_results()

def measurement_results():
    return "Hello"
