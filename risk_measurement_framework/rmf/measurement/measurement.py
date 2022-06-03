def mapping(low_l: list(), high_l: list()) -> list:

    if low_l is None:
        raise IndexError('Low-level attributes cannot be empty!')
    elif high_l is None:
        raise IndexError('High-Level attributes cannot be empty!')

    base_mea_raw = list()
    base_measures = list()

    for low_item in low_l:
        base_mea_raw.append(low_item)
        base_measures.append(low_item)

    for high_item in high_l:
        base_mea_raw.append(low_item)
        base_measures.append(low_item)

    return base_mea_raw, base_measures

def measurement_functions(base_measures: list()) -> list:

    def __ml_metrics():

        return metrics

    def __attack_steps():

        return steps

    def __extent_of_damage():

        return dmg

    ml_metrics = __ml_metrics()
    attack_steps = __attack_steps()
    dmg = __extent_of_damage()

    derived_measures = [ml_metrics, attack_steps, dmg]
    return derived_measures

def analytical_model(base_mea_raw: list(), derived_measures: list()):

def decision_criteria(*indicator) -> float:
