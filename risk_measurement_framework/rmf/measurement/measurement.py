from keras import backend as K

def mapping(low_l: list(), high_l: list()) -> list:

    if low_l is None:
        raise IndexError('Low-level attributes cannot be empty!')
    elif high_l is None:
        raise IndexError('High-Level attributes cannot be empty!')

    base_mea_raw = list()
    base_measures = list()

    for i, j in low_l.items():
        if j is "Raw":
            base_mea_raw.append(low_item)
        elif j is "Mapping":
            base_measures.append(low_item)

    return base_mea_raw, base_measures

def measurement_functions(base_measures: list(), y_true, y_pred) -> list:

    def __ml_metrics(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        return precision, recall, f1

    def __attack_steps(base_measures):

        return steps

    def __extent_of_damage(base_measures):

        return dmg

    precision, recall, f1 = __ml_metrics(y_true, y_pred)
    attack_steps = __attack_steps(base_measures)
    dmg = __extent_of_damage(base_measures)

    derived_measures = [precision, recall, f1, attack_steps, dmg]
    return derived_measures

def analytical_model(base_mea_raw: list(), derived_measures: list()):

    def __calc_eff():
        #base_measurs = [for base_mea in base_mea_raw if base_mea in ]
        #derived_measure = [for deri_mea in derived_measures if deri_mea in ]
        return "lol"

    def __calc_extent():
        for base_mea in base_mea_raw:
            print("Hello World")

        for der_mea in derived_measures:
            print("Hello World")

    effort = __calc_eff()
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
