"""Modelo de comidas: ¿qué tanto predicen los macronutrientes la respuesta glucémica?

Entrada: ``test.csv``.
Salidas: ``comidas_modelo.csv`` (lo usa ``modelo_general.py``) y ``probabilidades_Mcomidas.xlsx``.
"""

import numpy as np
from common import (
    ID_COLUMNS,
    LABEL_COLUMN,
    bin_nutrients_into_quartiles,
    load_labeled_meals,
    random_train_test_split,
    run_one_vs_rest,
)

MACRONUTRIENTS = ["hc_total", "fiber_total", "lipids_total", "protein_total"]


def build_meal_features(meals):
    """Macronutrientes de cada comida y las proporciones de carbohidratos y fibra sobre el total."""
    features = meals[MACRONUTRIENTS + ID_COLUMNS + [LABEL_COLUMN]].copy()
    features = features.dropna(subset=[LABEL_COLUMN])

    features["total_gr"] = features[MACRONUTRIENTS].sum(axis=1)
    features["Proporcion_hc"] = features["hc_total"] * 100 / features["total_gr"]
    # Proporción de fibra con respecto al total de lo que se consume.
    features["Proporcion_fiber"] = features["fiber_total"] * 100 / features["total_gr"]
    # Proporción de fibra y carbohidratos juntos con respecto al total.
    features["total_fib_carb"] = features["fiber_total"] + features["hc_total"]
    features["porc_fibra_carb"] = features["total_fib_carb"] * 100 / features["total_gr"]

    # Sin gramos registrados no se pueden calcular proporciones.
    return features.loc[features["total_gr"] > 0.0]


def main():
    np.random.seed(42)
    meal_features = build_meal_features(load_labeled_meals("test.csv"))
    meal_features.to_csv("comidas_modelo.csv", index=False)

    train, test = random_train_test_split(meal_features)
    helper_columns = ID_COLUMNS + ["total_gr", "total_fib_carb"]
    train = bin_nutrients_into_quartiles(train).drop(helper_columns, axis=1)
    test = bin_nutrients_into_quartiles(test).drop(helper_columns, axis=1)

    run_one_vs_rest(train, test, excel_path="probabilidades_Mcomidas.xlsx")


if __name__ == "__main__":
    main()
