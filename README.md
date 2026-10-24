# Respuesta glucémica a las comidas (iAUC) · C3, UNAM

Este repositorio tiene el código y los resultados de mi servicio social en el **Centro de Ciencias de la Complejidad (C3) de la UNAM**, dentro de un estudio de nutrición personalizada.

La pregunta principal fue: **¿qué hace que una comida suba mucho o poco la glucosa de una persona?** Para responderla se analizaron tres tipos de información:

- **La comida:** carbohidratos, fibra, lípidos y proteína.
- **La persona:** edad, sexo, estatura, circunferencia de cintura, HbA1c, triglicéridos, colesterol LDL/HDL, PCR, presión arterial, entre otras.
- **La microbiota intestinal:** abundancia de bacterias a nivel **filo** y a nivel **género**.

## Datos

- Cada participante usó un **monitor continuo de glucosa (CGM)** y registró sus comidas en una app (con foto en muchos casos).
- El estudio tuvo **tres visitas** (etapas `e1`, `e2` y `e3`). En cada una se tomaron los datos clínicos y una muestra de microbiota.
- *test.csv* tiene una fila por comida (2,811 comidas de 141 pacientes) con sus macronutrientes, la hora en que se reportó y las lecturas de glucosa antes y después de comer.
- Los datos clínicos y de microbiota se descargan de la API del estudio (`nutricion.c3.unam.mx`), que requiere acceso a la red del C3.

## Metodología

**1. Respuesta glucémica de cada comida.** Para cada comida se calcula la diferencia entre el área bajo la curva de glucosa **después** de comer y el doble del área **antes** de comer (la ventana basal dura la mitad que la posprandial). Solo se usan las comidas con diferencia positiva.

**2. Clases de respuesta.** Las comidas se separan en tres clases según esa diferencia: **Baja** (hasta 2,037), **Media** (2,037 a 4,074) y **Alta** (4,074 a 13,580).

**3. Selección de variables con épsilon.** Para cada valor de cada variable (por ejemplo, *HbA1c = Prediabetes*) y cada clase se calcula:

$$\varepsilon = \frac{n_{cx} - n_x \cdot P(C)}{\sqrt{n_x \cdot P(C)\,(1 - P(C))}}$$

donde $n_x$ es el número de comidas con ese valor, $n_{cx}$ las que además son de la clase $C$ y $P(C)$ la proporción de comidas de esa clase. Un **ε > 2** indica que esa combinación aparece más de lo que se esperaría por azar. Solo las variables con ε alto pasan a los modelos (ε > 1.8 para pacientes y ε > 2 para microbiota).

**4. Modelo bayesiano.** Para cada clase se entrena un clasificador bayesiano ingenuo de esa clase contra el resto. El score de una comida es

$$\text{score} = \log\frac{P(C)}{P(\neg C)} + \sum_i \log\frac{P(x_i \mid C)}{P(x_i \mid \neg C)}$$

Las variables numéricas se discretizan antes: los datos clínicos con sus rangos de referencia (por ejemplo, HbA1c normal, prediabetes o diabetes), la microbiota en deciles y los nutrientes en cuartiles. Las comidas se separan al azar en **70% entrenamiento y 30% prueba**, con semilla 42.

**5. Evaluación por deciles.** Las comidas de prueba se ordenan por score y se agrupan en deciles (D1 = score más bajo, D10 = más alto). Si el modelo funciona, los deciles altos deben tener más comidas de la clase (**Reales**) que los bajos.

Se entrenaron cinco modelos: **comidas**, **pacientes**, **microbiota (filo)**, **microbiota (género)** y uno **general** con todo junto.

## Estructura

```
Codigo_modelos/
├── common.py                    funciones compartidas: carga de datos, clases de iAUC, discretización,
│                                modelo bayesiano y tabla por deciles
├── epsilon_caracteristicas.py   Paso 1: épsilon de pacientes y microbiota → e_pats.csv, e_filo.csv, e_genero.csv
├── modelo_comidas.py            modelo con macronutrientes      → comidas_modelo.csv, probabilidades_Mcomidas.xlsx
├── modelo_pacientes.py          modelo con variables clínicas   → pats_modelo.csv, probabilidades_Mpats.xlsx
├── modelo_mic_filo.py           modelo con microbiota (filo)    → microbiota_modelo.csv, probabilidades_Mfilo.xlsx
├── modelo_mic_genero.py         modelo con microbiota (género)  → resultados en pantalla
├── modelo_general.py            modelo con todo junto           → probabilidades_Mgeneral.xlsx
├── test.csv                     comidas con sus lecturas de glucosa
└── archivos_utilizados/         archivos intermedios ya generados (para no depender de la API)
Resultados_Prob/                 probabilidades de cada modelo, una hoja por clase (Alta, Media, Baja)
```

## Instrucciones para ejecutar el código

Se necesita Python 3.10 o más reciente:

```bash
pip install -r requirements.txt
cd Codigo_modelos
```

El archivo principal es *test.csv*: es el único que se necesita en **TODOS** los códigos y el único que se debe tener **OBLIGATORIAMENTE** (o uno con la misma estructura).

Para correr los modelos se pueden copiar los archivos de **archivos_utilizados** a `Codigo_modelos/`, o generarlos así:

**Paso 1** - Correr *epsilon_caracteristicas.py*, que crea *e_pats.csv*, *e_filo.csv* y *e_genero.csv*. Cada modelo usa estos archivos para saber qué variables incluir.<br>
**Paso 2** - Correr en cualquier orden *modelo_comidas.py*, *modelo_pacientes.py*, *modelo_mic_filo.py* y *modelo_mic_genero.py*.<br>
**Paso 3** - Correr *modelo_general.py*, que junta las salidas del paso 2:
  - *modelo_comidas.py* **genera** *comidas_modelo.csv*
  - *modelo_pacientes.py* **genera** *pats_modelo.csv*
  - *modelo_mic_filo.py* **genera** *microbiota_modelo.csv*

Los pasos 1 y 2 (excepto *modelo_comidas.py*) consultan la API del estudio. *modelo_comidas.py* y *modelo_general.py* corren solo con los archivos del repositorio.

## Cómo leer los resultados

Cada script imprime, para cada clase, una tabla como esta (modelo de comidas, clase **Baja**):

|             | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | Total |
|-------------|----|----|----|----|----|----|----|----|----|-----|-------|
| Reales      | 37 | 34 | 34 | 36 | 34 | 32 | 30 | 29 | 37 | 41  | 344   |
| Complemento | 21 | 23 | 23 | 23 | 22 | 25 | 28 | 28 | 20 | 17  | 230   |

- **Reales:** comidas de prueba que sí son de la clase.
- **Complemento:** comidas de prueba de las otras clases.

Un buen modelo concentra las comidas **Reales** en los deciles altos (D9, D10). Los archivos de *Resultados_Prob* guardan, para cada variable y categoría, $P(C)$, $P(x_i \mid C)$, $P(x_i \mid \neg C)$ y los conteos $n_x$, $n_c$ y $n_{cx}$, con los que se puede ver qué categorías empujan una comida hacia cada clase.
