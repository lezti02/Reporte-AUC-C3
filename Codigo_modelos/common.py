"""Funciones compartidas por todos los modelos del reporte iAUC (C3, UNAM).

Antes, cada script repetía la carga de datos, la clasificación de la respuesta glucémica,
el modelo bayesiano y la tabla por deciles. Ahora viven aquí una sola vez, para que los
cinco modelos usen exactamente la misma lógica.

Los nombres de las columnas de los datos (``iAUC_reportada``, ``etapas``, ``patient``...) se
conservan tal como vienen de la base de datos y de los CSV ya generados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

API_URL = "https://nutricion.c3.unam.mx/nd"
VISITS = [1, 2, 3]

# Columna objetivo y sus clases.
LABEL_COLUMN = "iAUC_reportada"
IAUC_CLASSES = ["Baja", "Media", "Alta"]
# Límites de iAUC (mg/dL·min) que se consideraron para separar las comidas en las tres clases.
IAUC_BINS = [-13, 2036.979, 4073.958, 13579.861]

# Orden en que se reportan los resultados y orden de las hojas del Excel.
REPORT_ORDER = ["Alta", "Baja", "Media"]
EXCEL_SHEET_ORDER = ["Alta", "Media", "Baja"]

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
ID_COLUMNS = ["patient", "etapas"]

# Probabilidad que se usa cuando una categoría no aparece en el entrenamiento (evita log(0)).
MISSING_PROBABILITY = 1e-6

NUTRIENT_COLUMNS = ["hc_total", "fiber_total", "lipids_total", "protein_total",
                    "Proporcion_hc", "Proporcion_fiber", "porc_fibra_carb"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]

WAIST_LABELS = ["Bajo Riesgo", "Riesgo Elevado", "Riesgo Muy Elevado"]
WAIST_BINS_BY_SEX = {"Female": [0, 80, 88, 1000], "Male": [0, 94, 102, 1000]}
BLOOD_PRESSURE_LABELS = ["Baja", "Normal", "Elevada", "Hipertension 1", "Hipertension 2"]


# --------------------------------------------------------------------------- datos de la API
def fetch_visit_patient_data(visit: int) -> pd.DataFrame:
    """Datos clínicos de todos los pacientes en una visita (``etapas`` = 'e1', 'e2' o 'e3').

    Incluye también, como columnas, el incremento de AUC de cada alimento controlado de la
    curva de tolerancia.
    """
    url = f"{API_URL}/visits/{visit}/patients-data/"
    records = requests.get(url).json()["patient_data"]
    tolerance_auc = [
        {measure["controlled_food"]: measure["increment_auc"] for measure in record["tolerance_curve_measure"]}
        for record in records
    ]
    patients = pd.json_normalize(records)
    patients = patients.drop(columns=["tolerance_curve_measure", "sample.time_sample_was_taken"])
    patients = patients.rename(columns=lambda name: name.split(".")[1] if "." in name else name)
    patients = patients.rename(columns={"id": "patient"})
    patients["etapas"] = f"e{visit}"
    return pd.concat([patients, pd.DataFrame(tolerance_auc)], axis=1)


def fetch_all_visits_patient_data() -> pd.DataFrame:
    """Datos clínicos de las tres visitas, sin columnas repetidas."""
    visits = []
    for visit in VISITS:
        visit_data = fetch_visit_patient_data(visit).reset_index(drop=True)
        visits.append(visit_data.loc[:, ~visit_data.columns.duplicated()])
    return pd.concat(visits, axis=0)


def fetch_microbiota_by_visit(taxonomic_level: int, visit: int) -> pd.DataFrame:
    """Abundancia de cada organismo por paciente en una visita.

    ``taxonomic_level`` sigue la numeración de la API: 2 = filo, 6 = género.
    Regresa una tabla con un renglón por paciente (índice ``patient_id``), una columna por
    organismo y la columna ``visita``.
    """
    url = f"{API_URL}/microbiota/{taxonomic_level}/organisms-quantification/{visit}"
    quantification = pd.DataFrame(requests.get(url).json())
    quantification = quantification.set_index(["organism", "patient_id"])
    # La base de datos tiene análisis duplicados, por eso se suman por organismo y paciente.
    summed = quantification.groupby([pd.Grouper(level="organism"), pd.Grouper(level="patient_id")]).sum()
    table = summed.unstack(level=0).fillna(0)
    table = table.drop("quantification", axis=1)
    table.columns = table.columns.get_level_values(-1)
    table["visita"] = f"e{visit}"
    return table


def fetch_microbiota_all_visits(taxonomic_level: int) -> pd.DataFrame:
    """Microbiota de las tres visitas en una sola tabla, con columnas ``patient`` y ``etapas``."""
    visits = [fetch_microbiota_by_visit(taxonomic_level, visit).reset_index() for visit in VISITS]
    microbiota = pd.concat(visits, axis=0)
    return microbiota.rename(columns={"visita": "etapas", "patient_id": "patient"})


# ------------------------------------------------------------------ comidas y clases de iAUC
def load_labeled_meals(meals_path: str = "test.csv") -> pd.DataFrame:
    """Lee las comidas y les asigna su clase de respuesta glucémica (Baja, Media o Alta).

    La respuesta de cada comida es la diferencia entre el AUC posprandial y el doble del AUC
    basal (la ventana basal dura la mitad que la posprandial). Solo se conservan las comidas
    con diferencia positiva. Las comidas fuera de ``IAUC_BINS`` quedan sin clase (NaN).
    """
    meals = pd.read_csv(meals_path)
    meals["reportada_PPandrial_Basal_diferencia"] = (
        meals["reportada_PPandrial_auc"] - meals["reportada_basal_auc"] * 2
    )
    positive = meals[meals["reportada_PPandrial_Basal_diferencia"] >= 0].copy().reset_index(drop=True)
    positive[LABEL_COLUMN] = pd.cut(positive["reportada_PPandrial_Basal_diferencia"], bins=IAUC_BINS,
                                    right=True, labels=IAUC_CLASSES)
    return positive


def random_train_test_split(data: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """Separa los renglones al azar en entrenamiento (≈70%) y prueba.

    Usa el generador global de NumPy, así que el resultado depende de ``np.random.seed`` y del
    número de renglones: con la semilla 42 se obtiene la misma partición que en el reporte.
    """
    is_train = np.random.uniform(0, 1, len(data)) <= train_fraction
    return data[is_train].copy(), data[~is_train].copy()


# ------------------------------------------------------------------ discretización de variables
def bin_waist_by_sex(data: pd.DataFrame) -> pd.Series:
    """Riesgo por circunferencia de cintura, con puntos de corte distintos para mujeres y hombres."""
    waist = data["waist_circumference"].astype(object)
    for sex, bins in WAIST_BINS_BY_SEX.items():
        is_sex = data["sex"] == sex
        waist[is_sex] = pd.cut(data.loc[is_sex, "waist_circumference"].astype(float), bins=bins,
                               labels=WAIST_LABELS).astype(object)
    return waist


def bin_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convierte las variables clínicas en categorías con los rangos clínicos de referencia."""
    data["age"] = pd.cut(data["age"], bins=[0, 29, 39, 49, 59, 130], right=False, labels=None)
    data["waist_circumference"] = bin_waist_by_sex(data)
    data["tag"] = pd.cut(data["tag"], bins=[0, 150, 200, 500, 2000], right=False,
                         labels=["Normal", "Limite Alto", "Alto", "Muy Alto"])
    data["hdl"] = pd.cut(data["hdl"], bins=[0, 60, 1000], right=False, labels=["Bajo", "Normal"])
    data["height"] = pd.qcut(data["height"], q=10, labels=None)
    data["pas"] = pd.cut(data["pas"], bins=[0, 90, 120, 129, 139, 500], labels=BLOOD_PRESSURE_LABELS)
    data["pad"] = pd.cut(data["pad"], bins=[0, 60, 80, 89, 99, 500], labels=BLOOD_PRESSURE_LABELS)
    data["pcr"] = pd.cut(data["pcr"], bins=[0, 3, 10, 100, 500, 1000], right=False,
                         labels=["Normal", "Aumento Leve", "Aumento Moderado", "Aumento Alto", "Aumento Muy Alto"])
    data["hba1c"] = pd.cut(data["hba1c"], bins=[0, 5.7, 6.4, 100], right=False,
                           labels=["Normal", "Prediabetes", "Diabetes"])
    data["ldl"] = pd.cut(data["ldl"], bins=[0, 100, 130, 160, 190, 1000], right=False,
                         labels=["Optimo", "Casi optimo", "Limite Alto", "Alto", "Muy alto"])
    return data


def bin_into_deciles(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parte cada columna en deciles (menos si hay valores repetidos, p. ej. muchos ceros)."""
    for column in columns:
        data[column] = pd.qcut(data[column], q=10, labels=None, duplicates="drop")
    return data


def bin_nutrients_into_quartiles(data: pd.DataFrame) -> pd.DataFrame:
    """Parte cada nutriente y proporción de ``NUTRIENT_COLUMNS`` en cuartiles Q1–Q4."""
    for nutrient in NUTRIENT_COLUMNS:
        data[nutrient] = pd.qcut(data[nutrient], q=4, labels=QUARTILE_LABELS)
    return data


def feature_columns(data: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Columnas de ``data`` que no están en ``exclude``, en su orden original."""
    return [column for column in data.columns if column not in exclude]


# ---------------------------------------------------------------------- modelo bayesiano
def train_naive_bayes(train: pd.DataFrame, target_class: str, label_column: str = LABEL_COLUMN) -> dict:
    """Entrena un clasificador bayesiano ingenuo de una clase contra el resto.

    Para cada variable guarda P(x_i | C) y P(x_i | no C) por categoría, además de las
    probabilidades a priori P(C) y P(no C).
    """
    features = train.drop(label_column, axis=1)
    labels = train[label_column]
    in_class = train[labels == target_class]
    not_in_class = train[labels != target_class]

    prior = (labels == target_class).mean()
    return {
        "prior": prior,
        "prior_complement": 1 - prior,
        "likelihood": {col: in_class[col].value_counts(normalize=True).to_dict() for col in features.columns},
        "likelihood_complement": {col: not_in_class[col].value_counts(normalize=True).to_dict()
                                  for col in features.columns},
        "features": list(features.columns),
        "target_class": target_class,
    }


def score_naive_bayes(data: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Agrega la columna ``Score_<clase>`` con el log-momio bayesiano de cada renglón.

    score = log(P(C) / P(no C)) + Σ_i log(P(x_i | C) / P(x_i | no C))

    Una categoría que no se vio en el entrenamiento recibe probabilidad ``MISSING_PROBABILITY``.
    Si una categoría nunca aparece dentro de la clase, P(x_i | C) = 0 y el score es -inf.
    """
    scored = data.copy()
    score = np.log(model["prior"] / model["prior_complement"])
    for col in model["features"]:
        values = data[col].astype(object)
        p_in_class = values.map(model["likelihood"][col]).fillna(MISSING_PROBABILITY).astype(float)
        p_complement = values.map(model["likelihood_complement"][col]).fillna(MISSING_PROBABILITY).astype(float)
        p_complement = p_complement.replace(0, MISSING_PROBABILITY)
        with np.errstate(divide="ignore"):
            score = score + np.log(p_in_class / p_complement)
    scored["Score_" + model["target_class"]] = score
    return scored


def probability_table(model: dict, data: pd.DataFrame, label_column: str = LABEL_COLUMN) -> pd.DataFrame:
    """Tabla con las probabilidades del modelo y los conteos nx, nc y ncx de cada categoría.

    nx = renglones con la categoría, nc = renglones de la clase, ncx = renglones con ambas.
    """
    target_class = model["target_class"]
    in_class = data[label_column] == target_class
    rows = []
    for variable, likelihood in model["likelihood"].items():
        categories = set(likelihood) | set(model["likelihood_complement"][variable])
        for category in sorted(categories):
            has_category = data[variable] == category
            rows.append({
                "variable": variable,
                "cat": category,
                "p_c": model["prior"],
                "p_no_c": model["prior_complement"],
                "p_xi_C": likelihood.get(category, np.nan),
                "p_xi_noC": model["likelihood_complement"][variable].get(category, np.nan),
                "nx": has_category.sum(),
                "nc": in_class.sum(),
                "ncx": (has_category & in_class).sum(),
            })
    return pd.DataFrame(rows).set_index(["variable", "cat"])


def decile_table(scored: pd.DataFrame, target_class: str, drop_duplicate_edges: bool = False,
                 label_column: str = LABEL_COLUMN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordena las comidas de prueba por score, las agrupa en deciles y cuenta aciertos.

    Si el modelo funciona, los deciles altos (D10) deben tener más comidas de la clase
    ("Reales") que los bajos (D1). Los -inf se reemplazan por el mínimo finito menos 1 para
    que entren en el primer decil.

    ``drop_duplicate_edges`` permite menos de 10 grupos cuando muchos scores son iguales.

    Regresa ``(comidas_con_decil, tabla_de_conteos)``.
    """
    score_column = "Score_" + target_class
    ranked = scored[[label_column, score_column]].copy().sort_values(by=score_column)
    finite_min = ranked[score_column][ranked[score_column] != -np.inf].min()
    ranked[score_column] = ranked[score_column].replace(-np.inf, finite_min - 1)
    if drop_duplicate_edges:
        n_groups = pd.qcut(ranked[score_column], q=10, labels=None, duplicates="drop").nunique()
        ranked["Deciles"] = pd.qcut(ranked[score_column], q=10, duplicates="drop",
                                    labels=[f"D{i}" for i in range(1, n_groups + 1)])
    else:
        ranked["Deciles"] = pd.qcut(ranked[score_column], q=10, labels=[f"D{i}" for i in range(1, 11)])

    hits, misses = {}, {}
    for decile in ranked["Deciles"].unique():
        in_decile = ranked[ranked["Deciles"] == decile]
        hits[decile] = int((in_decile[label_column] == target_class).sum())
        misses[decile] = len(in_decile) - hits[decile]
    counts = pd.DataFrame([hits, misses], index=["Reales", "Complemento"])
    counts["Total"] = counts.sum(axis=1)
    return ranked, counts


def run_one_vs_rest(train: pd.DataFrame, test: pd.DataFrame, excel_path: str | None = None,
                    drop_duplicate_edges: bool = False, verbose: bool = True) -> dict:
    """Entrena un modelo por clase (Alta, Media, Baja), evalúa en prueba y guarda las probabilidades.

    Imprime, para cada clase, las comidas de prueba ordenadas por score y la tabla de aciertos
    por decil. Si se da ``excel_path``, guarda la tabla de probabilidades de cada clase en una
    hoja del Excel.
    """
    results = {}
    for target_class in REPORT_ORDER:
        model = train_naive_bayes(train, target_class)
        ranked, counts = decile_table(score_naive_bayes(test, model), target_class, drop_duplicate_edges)
        results[target_class] = {"model": model, "ranked": ranked, "counts": counts}
        if excel_path:
            results[target_class]["probabilities"] = probability_table(model, train)
        if verbose:
            print(f"-------------{target_class.upper()}----------------")
            print(ranked)
            print(counts)

    if excel_path:
        with pd.ExcelWriter(excel_path) as writer:
            for target_class in EXCEL_SHEET_ORDER:
                results[target_class]["probabilities"].to_excel(writer, sheet_name=target_class)
    return results


def run_microbiota_model(taxonomic_level: int, epsilons_path: str, features_csv_path: str | None = None,
                         excel_path: str | None = None, drop_duplicate_edges: bool = False) -> dict:
    """Modelo de microbiota a nivel filo o género.

    Usa solo los organismos que salieron relevantes en el análisis de épsilon, discretiza su
    abundancia en deciles y corre el modelo bayesiano para cada clase de iAUC.
    """
    np.random.seed(RANDOM_SEED)
    microbiota = fetch_microbiota_all_visits(taxonomic_level)
    relevant = pd.read_csv(epsilons_path)["Caracteristica"].unique().tolist()
    microbiota = microbiota[relevant + ID_COLUMNS].copy()
    if features_csv_path:
        microbiota.to_csv(features_csv_path, index=False)

    meals = load_labeled_meals("test.csv")[ID_COLUMNS + [LABEL_COLUMN]].dropna(subset=[LABEL_COLUMN])
    dataset = meals.merge(microbiota, on=ID_COLUMNS, how="left")

    train, test = random_train_test_split(dataset)
    organisms = feature_columns(dataset, exclude=[LABEL_COLUMN] + ID_COLUMNS)
    train = bin_into_deciles(train, organisms).drop(ID_COLUMNS, axis=1)
    test = bin_into_deciles(test, organisms).drop(ID_COLUMNS, axis=1)
    return run_one_vs_rest(train, test, excel_path=excel_path, drop_duplicate_edges=drop_duplicate_edges)
