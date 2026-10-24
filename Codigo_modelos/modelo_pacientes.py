"""Modelo de pacientes: ¿qué tanto predicen las variables clínicas la respuesta glucémica?

Entradas: ``test.csv``, ``e_pats.csv`` (de epsilon_caracteristicas.py) y la API del estudio.
Salidas: ``pats_modelo.csv`` (lo usa ``modelo_general.py``) y ``probabilidades_Mpats.xlsx``.
"""

import common
import numpy as np
import pandas as pd
from common import (
    ID_COLUMNS,
    LABEL_COLUMN,
    bin_clinical_features,
    load_labeled_meals,
    random_train_test_split,
    run_one_vs_rest,
)


def select_relevant_patient_features(patients, epsilons_path="e_pats.csv"):
    """Conserva solo las variables clínicas que salieron relevantes (|ε| alto) en el análisis de épsilon."""
    epsilons = pd.read_csv(epsilons_path)
    columns = epsilons["Caracteristica"].unique().tolist() + ID_COLUMNS
    return patients[columns].copy()


def main():
    np.random.seed(42)
    patient_features = select_relevant_patient_features(common.fetch_all_visits_patient_data())
    patient_features.to_csv("pats_modelo.csv", index=False)

    meals = load_labeled_meals("test.csv")[ID_COLUMNS + [LABEL_COLUMN]].dropna(subset=[LABEL_COLUMN])
    dataset = meals.merge(patient_features, on=ID_COLUMNS, how="left")

    train, test = random_train_test_split(dataset)
    train = bin_clinical_features(train).drop(ID_COLUMNS, axis=1)
    test = bin_clinical_features(test).drop(ID_COLUMNS, axis=1)

    run_one_vs_rest(train, test, excel_path="probabilidades_Mpats.xlsx")


if __name__ == "__main__":
    main()
