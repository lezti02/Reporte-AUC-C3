import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.integrate as integrate
import math

# función para obtener los datos de los pacientes en cada visita 
def readVisitas(visita):

    url_visitas = url_api + "/visits/" + str(visita) + "/patients-data/"
    datos = requests.get(url_visitas).json()["patient_data"]
    l1 = [ { cf["controlled_food"]: cf["increment_auc"] for cf in tolerance["tolerance_curve_measure"] } for tolerance in datos ]
    df1 = pd.json_normalize(datos)
    df1.drop("tolerance_curve_measure", inplace=True, axis=1)
    df1.drop('sample.time_sample_was_taken', inplace=True, axis=1)
    df1.rename(columns= lambda x: x.split(".")[1] if "." in x else x, inplace=True)
    df1.rename(columns={'id': 'patient'}, inplace = True)
    df1['etapas'] = 'e' + str(visita)
    return  pd.concat([df1, pd.DataFrame(l1)], axis=1)


def calcular_epsilons(df):

    resultados = []
    n = len(df)  # Total de muestras
    clases_auc = df['iAUC_reportada'].unique()  # ['Baja', 'Media', 'Alta']
    df_limpio = df.copy()
    df_limpio.drop(['patient','etapas','iAUC_reportada'], inplace = True, axis = 1)
    features = df_limpio.columns # Características a evaluar
    
    for feature in features:
        valores = df[feature].dropna().unique()  # Valores únicos de la característica 
        
        for valor in valores:
            # Número de muestras con la característica actual (nx)
            nx = len(df[df[feature] == valor])
            if nx == 0:
                continue  # Evitar división por cero
            
            for clase in clases_auc:
                # Número de muestras en la clase de AUC (nc)
                nc = len(df[df['iAUC_reportada'] == clase])
                # Número de muestras con el valor actual Y en la clase de AUC (ncx)
                ncx = len(df[(df[feature] == valor) & (df['iAUC_reportada'] == clase)])
                
                # Calcular épsilon
                if n == 0:
                    epsilon = 0.0
                else:
                    denominador = math.sqrt(nx * (nc / n) * (1 - (nc / n)))
                    epsilon = (ncx - ((nc/n)*nx))/denominador if denominador != 0 else 0.0  #formula para obtener epsilon
                    
                resultados.append({
                    'Caracteristica': feature,
                    'Valor': valor,
                    'Clase_AUC': clase,
                    'Epsilon': epsilon,
                    'nx': nx,
                    'nc': nc,
                    'ncx': ncx
                })
    
    return pd.DataFrame(resultados)


def microbiota_quanta_visit(taxa_level, visita):
    url_taxones = url_api + "/microbiota/" +  str(taxa_level) + "/organisms-quantification/" + str(visita)     
    lista = requests.get(url_taxones).json()
    taxones = lista
    dftax = pd.DataFrame(taxones)
    print(dftax.columns)
    print(dftax["patient_id"].unique())
    dftax.set_index(["organism","patient_id"], inplace=True)
    matrix = dftax.groupby([pd.Grouper(level="organism"), pd.Grouper(level="patient_id")]).sum() # esto es necesario solo porque la BD tiene análisis duplicados
    tabla = matrix.unstack(level=0).fillna(0)
    tabla.drop(("quantification"), axis=1, inplace=True)
    tabla.columns=tabla.columns.get_level_values(-1)
    tabla["visita"] = "e" + str(visita)
    return tabla


test = pd.read_csv('test.csv')   #dataset con los datos de todas las comidas consideradas

url_api = 'https://nutricion.c3.unam.mx/nd'

# diferencia entre auc( pospandrial - basal) reportada
test['reportada_PPandrial_Basal_diferencia'] = test['reportada_PPandrial_auc'] - (test['reportada_basal_auc']) * 2   

# df en el que solo estan los valores positivos de la diferencia
test_2 = test[test['reportada_PPandrial_Basal_diferencia'] >= 0].copy().reset_index(drop=True)    

# crear nueva columna rangos donde categorice los valores de las diferencias positivas
test_2['rangos'] = pd.cut(test_2['reportada_PPandrial_Basal_diferencia'], 20, right=True, labels=None)  

# df con las categorias que necesito para las caracteristicas de pacientes

pats_info = [readVisitas(visita).reset_index(drop=True) for visita in [1, 2, 3]]
for decil in range(len(pats_info)):
    pats_info[decil] = pats_info[decil].loc[:, ~pats_info[decil].columns.duplicated()]

pats_todas = pd.concat(pats_info, axis=0)

ensamble = test_2[['patient','etapas']].copy()
bins_rangos = [-13, 2036.979, 4073.958, 13579.861]
ensamble['iAUC_reportada'] = pd.cut(test_2['reportada_PPandrial_Basal_diferencia'], bins = bins_rangos, right=True, labels=['Baja', 'Media', 'Alta'])
#print(pats_info.columns)
pats = pats_todas[['patient','etapas','sex', 'age', 'height','weight', 'waist_circumference', 'imc', 'hba1c', 'tag', 'glucose', 'ct', 'ldl', 'hdl',
                  'pcr', 'alt', 'ast', 'homa', 'insuline', 'pas', 'pad', 'ipaq_class', 'stress_class','hadsa_class', 'hadsd_class']].copy()
  
pats['age'] = pd.cut(pats['age'], bins = [0,29,39,49,59,130], right=False, labels=None) 
pats['imc'] = pd.cut(pats['imc'], bins = [0, 18.5, 25, 30, 200], right=False, labels=['Bajo', 'Normal', 'Sobrepeso', 'Obesidad'])
pats['hba1c'] = pd.cut(pats['hba1c'], bins = [0,5.7, 6.4, 100 ], right=False, labels=['Normal', 'Prediabetes', 'Diabetes'])
pats['tag'] = pd.cut(pats['tag'], bins = [0, 150, 200, 500, 2000], right=False, labels=['Normal', 'Limite Alto', 'Alto', 'Muy Alto'])    
pats['glucose'] = pd.cut(pats['glucose'], bins = [0, 85, 95, 100, 110, 900], right=False, labels=['Bajo', 'Normal-1', 'Normal-2', 'Alto', 'Muy alto'])
pats['ct'] = pd.cut(pats['ct'], bins = [0, 50, 100, 150, 200, 250,1000], right=False, labels=['Bajo', 'Normal-1', 'Normal-2', 'Normal-3', 'Alto', 'Muy alto'])
pats['ldl'] = pd.cut(pats['ldl'], bins = [0, 100, 130, 160, 190, 1000], right=False, labels=['Optimo', 'Casi optimo', 'Limite Alto', 'Alto', 'Muy alto'])
pats['hdl'] = pats['hdl'] = pd.cut(pats['hdl'], bins = [0, 60, 1000], right=False, labels=["bajo", "normal"])
pats['pcr'] = pd.cut(pats['pcr'], bins = [0,3,10,100, 500, 1000], right=False, labels=['Normal', 'Aumento Leve', 'Aumento Moderado', 'Aumento Alto', 'Aumento Muy Alto'])
pats['alt'] = pd.cut(pats['alt'], bins = [0,4, 36, 500], right=False, labels=['Baja', 'Normal', 'Alta'])
pats['ast'] = pd.cut(pats['ast'], bins = [0,8, 33, 500], right=False, labels=['Baja', 'Normal', 'Alta'])
pats['homa'] = pd.cut(pats['homa'], bins = [0,1.96, 3, 100], right=False, labels=['Sin resistencia', 'Sospecha', 'Resistencia'])
pats['insuline'] = pd.cut(pats['insuline'], bins = [0,5,25,30,100], right=False, labels=['Bajo', 'Normal', 'Sospechoso', 'Alto'])


pats.loc[pats['sex'] == 'Female', 'waist_circumference'] = pd.cut(pats.loc[pats['sex'] == 'Female', 'waist_circumference'],
                                                                  bins=[0,80, 88, 1000], labels= ['Bajo Riesgo', 'Riesgo Elevado', 'Riesgo Muy Elevado'])
pats.loc[pats['sex'] == 'Male','waist_circumference'] = pd.cut(pats.loc[pats['sex'] == 'Male', 'waist_circumference'],
                                                                  bins=[0,94, 102, 1000], labels= ['Bajo Riesgo', 'Riesgo Elevado', 'Riesgo Muy Elevado'])

pats['weight'] = pd.cut(pats['weight'], bins = 7, labels = None)
pats['height'] = pd.cut(pats['height'], bins = 5, labels = None)
pats['pas'] = pd.cut(pats['pas'], bins = [0,90,120,129,139,500], labels = ['Baja','Normal','Elevada', 'Hipertension 1', 'Hipertension 2'])
pats['pad'] = pd.cut(pats['pad'], bins = [0,60,80,89,99,500], labels = ['Baja','Normal','Elevada', 'Hipertension 1', 'Hipertension 2'])


#print(pats)
pats_subset = pats[['patient','etapas', 'sex', 'age', 'weight', 'height', 'waist_circumference', 'imc', 'hba1c', 'tag','glucose', 'ct', 'ldl', 'hdl',
                    'pcr', 'alt', 'ast', 'homa', 'insuline', 'pas', 'pad', 'ipaq_class', 'stress_class','hadsa_class', 'hadsd_class']]  
ensamble_c = pd.merge(ensamble,pats_subset, on=['patient','etapas'], how='left')  #ensamble para las caracteristicas

# Sacar lo contenido en microbiota para filo
micros_filo = [ microbiota_quanta_visit(2, visita) for visita in [1,2,3]]
todas_filo = pd.concat(micros_filo, axis=0)
todas_filo["patient"] = todas_filo.index
todas_filo.rename(columns = {"visita":"etapas"}, inplace=True)


# Sacar lo contenido en microbiota para genero 
micros_genero = [ microbiota_quanta_visit(6, visita) for visita in [1,2,3]]
todas_genero = pd.concat(micros_genero, axis=0)
todas_genero["patient"] = todas_genero.index
todas_genero.rename(columns = {"visita":"etapas"}, inplace=True)

# print(ensamble.info())
# print(ensamble.describe(include='all'))

for i in ensamble:
    frecuencia_columna = ensamble[i].value_counts()
    print(frecuencia_columna)

ensamble_filo = pd.merge(ensamble,todas_filo, on=['patient','etapas'], how='left')  # ensamble para microbiota filo 
#print(ensamble2) 
ensamble_genero = pd.merge(ensamble,todas_genero, on=['patient','etapas'], how='left') 
print(ensamble_genero) 

resultados_epsilon = calcular_epsilons(ensamble_c)  #epsilones de caracteristicas pacientes
e_pats = resultados_epsilon.loc[resultados_epsilon['Epsilon'] > 1.8]
e_pats.to_csv('e_pats.csv')
# print(resultados_epsilon)




columnas = ensamble_filo.columns
lista = columnas.tolist()
elementos_eliminar = ['iAUC_reportada', 'patient', 'etapas']
for elemento in elementos_eliminar:
    while elemento in lista:
        lista.remove(elemento)
for org in lista:
    ensamble_filo[org] = pd.qcut(ensamble_filo[org], q=10,
            labels=None,
            duplicates='drop')


epsilon_filo = calcular_epsilons(ensamble_filo) #epsilones de microbiota filo
#print(epsilon_filo)
e_filo = epsilon_filo.loc[epsilon_filo['Epsilon'] > 2]
e_filo.to_csv('e_filo.csv')



columnas = ensamble_genero.columns
lista = columnas.tolist()
elementos_eliminar = ['iAUC_reportada', 'patient', 'etapas']
for elemento in elementos_eliminar:
    while elemento in lista:
        lista.remove(elemento)
for org in lista:
    ensamble_genero[org] = pd.qcut(ensamble_genero[org], q=10,
            labels=None,
            duplicates='drop')



epsilon_genero = calcular_epsilons(ensamble_genero) #epsilones microbiota genero 
e_genero = epsilon_genero.loc[epsilon_genero['Epsilon'] > 2]
e_genero.to_csv('e_genero.csv')















