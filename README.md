## Instrucciones para ejecutar el código
El archivo principal es *test.csv*, este es el único archivo que se necesita en **TODOS** los códigos y el único que se debe tener o descargar **OBLIGATORIAMENTE** o tener uno similar a este.

Para poder correr los códigos relacionados a los modelos, se puede o bien descargar los archivos dentro de la carpeta **archivos_utilizados** o: 

**Paso 1** - Correr el programa *epsilones_caracteristicas.py* de donde se crean los dataframes *[e_filo.csv, e_genero.csv y e_pats.csv]* los cuales se utilizan para correr cada uno de los modelos. <br>
**Paso 2** - Una vez hecho el paso 1 ya se pueden correr en cualquier orden el resto de los códigos con excepción del archivo *modelo_general.py*<br>
**Paso 3** - Si se desea correr el archivo *modelo_general.py* primero hay que correr los otros 3 modelos para obtener dataframes necesarios para poder hacer un merge y tener todos los datos agrupados en un solo dataframe.<br>
  - *modelo_comidas.py* **genera** *modelo_test.csv*<br>
  - *modelo_pacientes.py* **genera** *modelo_pacientes.csv*<br>
  - *modelo_mic_filo.py* **genera** *modelo_microbiota.csv*
