"""Paso 1: análisis de épsilon para elegir qué variables usar en cada modelo.

Para cada valor de cada variable (por ejemplo, "HbA1c = Prediabetes") y cada clase de iAUC,
épsilon mide qué tanto se aleja el número de comidas observadas del que se esperaría por azar:

    ε = (n_cx - n_x · P(C)) / sqrt(n_x · P(C) · (1 - P(C)))

Un ε > 2 indica que esa combinación aparece más de lo esperado (aprox. 95% de confianza).

Entradas: ``test.csv`` y la API del estudio (datos clínicos y microbiota).
Salidas: ``e_pats.csv``, ``e_filo.csv`` y ``e_genero.csv``, que usan los demás modelos.
"""

import math

import common
import pandas as pd
from common import (
    BLOOD_PRESSURE_LABELS,
    ID_COLUMNS,
    LABEL_COLUMN,
    VISITS,
    bin_into_deciles,
    bin_waist_by_sex,
    feature_columns,
    load_labeled_meals,
)

PATIENT_EPSILON_THRESHOLD = 1.8
MICROBIOTA_EPSILON_THRESHOLD = 2
PHYLUM_LEVEL, GENUS_LEVEL = 2, 6

PATIENT_FEATURES = ["sex", "age", "weight", "height", "waist_circumference", "imc", "hba1c", "tag", "glucose", "ct",
                    "ldl", "hdl", "pcr", "alt", "ast", "homa", "insuline", "pas", "pad", "ipaq_class", "stress_class",
                    "hadsa_class", "hadsd_class"]


def compute_epsilons(data, label_column=LABEL_COLUMN):
    """Calcula ε para cada (variable, valor, clase de iAUC).

    ``data`` debe tener las columnas ``patient``, ``etapas``, la columna de clase y las
    variables ya discretizadas. Regresa una tabla con ``Caracteristica``, ``Valor``,
    ``Clase_AUC``, ``Epsilon``, ``nx`` (comidas con ese valor), ``nc`` (comidas de la clase)
    y ``ncx`` (comidas con ambos).
    """
    rows = []
    n_total = len(data)
    classes = data[label_column].unique()
    features = data.drop(["patient", "etapas", label_column], axis=1).columns

    for feature in features:
        for value in data[feature].dropna().unique():
            has_value = data[feature] == value
            n_x = int(has_value.sum())
            if n_x == 0:
                continue
            for auc_class in classes:
                in_class = data[label_column] == auc_class
                n_c = int(in_class.sum())
                n_cx = int((has_value & in_class).sum())
                if n_total == 0:
                    epsilon = 0.0
                else:
                    p_class = n_c / n_total
                    denominator = math.sqrt(n_x * p_class * (1 - p_class))
                    epsilon = (n_cx - p_class * n_x) / denominator if denominator != 0 else 0.0
                rows.append({"Caracteristica": feature, "Valor": value, "Clase_AUC": auc_class, "Epsilon": epsilon,
                             "nx": n_x, "nc": n_c, "ncx": n_cx})
    return pd.DataFrame(rows)


def bin_patient_features_for_epsilon(patients):
    """Discretiza todas las variables clínicas con sus rangos de referencia.

    Usa más variables que ``common.bin_clinical_features`` porque aquí todavía no se sabe
    cuáles son relevantes.
    """
    patients = patients[ID_COLUMNS + PATIENT_FEATURES].copy()
    patients["age"] = pd.cut(patients["age"], bins=[0, 29, 39, 49, 59, 130], right=False, labels=None)
    patients["imc"] = pd.cut(patients["imc"], bins=[0, 18.5, 25, 30, 200], right=False,
                             labels=["Bajo", "Normal", "Sobrepeso", "Obesidad"])
    patients["hba1c"] = pd.cut(patients["hba1c"], bins=[0, 5.7, 6.4, 100], right=False,
                               labels=["Normal", "Prediabetes", "Diabetes"])
    patients["tag"] = pd.cut(patients["tag"], bins=[0, 150, 200, 500, 2000], right=False,
                             labels=["Normal", "Limite Alto", "Alto", "Muy Alto"])
    patients["glucose"] = pd.cut(patients["glucose"], bins=[0, 85, 95, 100, 110, 900], right=False,
                                 labels=["Bajo", "Normal-1", "Normal-2", "Alto", "Muy alto"])
    patients["ct"] = pd.cut(patients["ct"], bins=[0, 50, 100, 150, 200, 250, 1000], right=False,
                            labels=["Bajo", "Normal-1", "Normal-2", "Normal-3", "Alto", "Muy alto"])
    patients["ldl"] = pd.cut(patients["ldl"], bins=[0, 100, 130, 160, 190, 1000], right=False,
                             labels=["Optimo", "Casi optimo", "Limite Alto", "Alto", "Muy alto"])
    patients["hdl"] = pd.cut(patients["hdl"], bins=[0, 60, 1000], right=False, labels=["bajo", "normal"])
    patients["pcr"] = pd.cut(patients["pcr"], bins=[0, 3, 10, 100, 500, 1000], right=False,
                             labels=["Normal", "Aumento Leve", "Aumento Moderado", "Aumento Alto", "Aumento Muy Alto"])
    patients["alt"] = pd.cut(patients["alt"], bins=[0, 4, 36, 500], right=False, labels=["Baja", "Normal", "Alta"])
    patients["ast"] = pd.cut(patients["ast"], bins=[0, 8, 33, 500], right=False, labels=["Baja", "Normal", "Alta"])
    patients["homa"] = pd.cut(patients["homa"], bins=[0, 1.96, 3, 100], right=False,
                              labels=["Sin resistencia", "Sospecha", "Resistencia"])
    patients["insuline"] = pd.cut(patients["insuline"], bins=[0, 5, 25, 30, 100], right=False,
                                  labels=["Bajo", "Normal", "Sospechoso", "Alto"])
    patients["waist_circumference"] = bin_waist_by_sex(patients)
    patients["weight"] = pd.cut(patients["weight"], bins=7, labels=None)
    patients["height"] = pd.cut(patients["height"], bins=5, labels=None)
    patients["pas"] = pd.cut(patients["pas"], bins=[0, 90, 120, 129, 139, 500], labels=BLOOD_PRESSURE_LABELS)
    patients["pad"] = pd.cut(patients["pad"], bins=[0, 60, 80, 89, 99, 500], labels=BLOOD_PRESSURE_LABELS)
    return patients


def load_microbiota(taxonomic_level):
    """Microbiota de las tres visitas, con ``patient`` y ``etapas`` como columnas."""
    visits = [common.fetch_microbiota_by_visit(taxonomic_level, visit) for visit in VISITS]
    microbiota = pd.concat(visits, axis=0)
    microbiota["patient"] = microbiota.index
    return microbiota.rename(columns={"visita": "etapas"})


def microbiota_epsilons(labeled_meals, taxonomic_level):
    """ε de cada organismo, con su abundancia partida en deciles."""
    dataset = labeled_meals.merge(load_microbiota(taxonomic_level), on=ID_COLUMNS, how="left")
    organisms = feature_columns(dataset, exclude=[LABEL_COLUMN] + ID_COLUMNS)
    return compute_epsilons(bin_into_deciles(dataset, organisms))


def main():
    labeled_meals = load_labeled_meals("test.csv")[ID_COLUMNS + [LABEL_COLUMN]]

    patients = bin_patient_features_for_epsilon(common.fetch_all_visits_patient_data())
    patient_epsilons = compute_epsilons(labeled_meals.merge(patients, on=ID_COLUMNS, how="left"))
    patient_epsilons.loc[patient_epsilons["Epsilon"] > PATIENT_EPSILON_THRESHOLD].to_csv("e_pats.csv")

    for level, output in [(PHYLUM_LEVEL, "e_filo.csv"), (GENUS_LEVEL, "e_genero.csv")]:
        epsilons = microbiota_epsilons(labeled_meals, level)
        epsilons.loc[epsilons["Epsilon"] > MICROBIOTA_EPSILON_THRESHOLD].to_csv(output)


if __name__ == "__main__":
    main()
