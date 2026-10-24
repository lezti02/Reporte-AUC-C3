"""Modelo general: comidas, microbiota (filo) y variables clínicas juntas.

Necesita las salidas de los otros modelos:
``comidas_modelo.csv`` (modelo_comidas.py), ``microbiota_modelo.csv`` (modelo_mic_filo.py)
y ``pats_modelo.csv`` (modelo_pacientes.py).
Salida: ``probabilidades_Mgeneral.xlsx``.
"""

import numpy as np
import pandas as pd
from common import (
    ID_COLUMNS,
    LABEL_COLUMN,
    bin_clinical_features,
    bin_into_deciles,
    bin_nutrients_into_quartiles,
    feature_columns,
    random_train_test_split,
    run_one_vs_rest,
)


def prepare_features(data, microbiota_columns):
    """Discretiza las variables clínicas, la microbiota (deciles) y los nutrientes (cuartiles)."""
    data = bin_clinical_features(data)
    data = bin_into_deciles(data, microbiota_columns)
    return bin_nutrients_into_quartiles(data)


def main():
    np.random.seed(42)
    meals = pd.read_csv("comidas_modelo.csv")
    microbiota = pd.read_csv("microbiota_modelo.csv")
    patients = pd.read_csv("pats_modelo.csv")

    dataset = meals.merge(microbiota, how="left", on=ID_COLUMNS)
    dataset = dataset.merge(patients, how="left", on=ID_COLUMNS)
    dataset = dataset.dropna(subset=[LABEL_COLUMN])

    train, test = random_train_test_split(dataset)
    microbiota_columns = feature_columns(microbiota, exclude=ID_COLUMNS)
    helper_columns = ID_COLUMNS + ["total_gr", "total_fib_carb"]
    train = prepare_features(train, microbiota_columns).drop(helper_columns, axis=1)
    test = prepare_features(test, microbiota_columns).drop(helper_columns, axis=1)

    run_one_vs_rest(train, test, excel_path="probabilidades_Mgeneral.xlsx")


if __name__ == "__main__":
    main()
