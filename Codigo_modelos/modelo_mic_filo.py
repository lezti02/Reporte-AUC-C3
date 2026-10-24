"""Modelo de microbiota a nivel filo.

Entradas: ``test.csv``, ``e_filo.csv`` (de epsilon_caracteristicas.py) y la API del estudio.
Salidas: ``microbiota_modelo.csv`` (lo usa ``modelo_general.py``) y ``probabilidades_Mfilo.xlsx``.

Como varios pacientes comparten la misma abundancia, muchos scores se repiten y se permiten
menos de 10 deciles.
"""

from common import run_microbiota_model

PHYLUM_LEVEL = 2

if __name__ == "__main__":
    run_microbiota_model(PHYLUM_LEVEL, "e_filo.csv", features_csv_path="microbiota_modelo.csv",
                         excel_path="probabilidades_Mfilo.xlsx", drop_duplicate_edges=True)
