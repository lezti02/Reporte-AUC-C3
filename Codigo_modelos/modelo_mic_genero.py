"""Modelo de microbiota a nivel género.

Entradas: ``test.csv``, ``e_genero.csv`` (de epsilon_caracteristicas.py) y la API del estudio.
A nivel género muchos pacientes tienen abundancia cero, así que varios scores se repiten y se
permiten menos de 10 deciles. Los resultados se imprimen en pantalla.
"""

from common import run_microbiota_model

GENUS_LEVEL = 6

if __name__ == "__main__":
    run_microbiota_model(GENUS_LEVEL, "e_genero.csv", drop_duplicate_edges=True)
