import pandas as pd
import numpy as np
import math
import requests

np.random.seed(42)


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



url_api = 'https://nutricion.c3.unam.mx/nd'


pats_info = [readVisitas(visita).reset_index(drop=True) for visita in [1, 2, 3]]
for decil in range(len(pats_info)):
    pats_info[decil] = pats_info[decil].loc[:, ~pats_info[decil].columns.duplicated()]

pats_todas = pd.concat(pats_info, axis=0)

epsilones = pd.read_csv('e_pats.csv')
#print(epsilones)
columnas = epsilones['Caracteristica'].unique()
lista = columnas.tolist()
lista.extend(['patient', 'etapas'])
print(lista)
pats_modelo = pats_todas[lista].copy()
print(pats_modelo)
pats_modelo.to_csv('pats_modelo.csv')

df1 = pd.read_csv('test.csv')   #dataset con los datos de todas las comidas consideradas

# diferencia entre auc( pospandrial - basal) reportada
df1['reportada_PPandrial_Basal_diferencia'] = df1['reportada_PPandrial_auc'] - (df1['reportada_basal_auc']) * 2   

# df en el que solo estan los valores positivos de la diferencia
df2 = df1[df1['reportada_PPandrial_Basal_diferencia'] >= 0].copy().reset_index(drop=True)    

# crear nueva columna rangos donde categorice los valores de las diferencias positivas
df2['rangos'] = pd.cut(df2['reportada_PPandrial_Basal_diferencia'], 20, right=True, labels=None)  
ensamble = df2[['patient','etapas']].copy()
bins_rangos = [-13, 2036.979, 4073.958, 13579.861]   # los rangos que se consideraron para separar las comidas
ensamble['iAUC_reportada'] = pd.cut(df2['reportada_PPandrial_Basal_diferencia'], bins = bins_rangos, right=True, labels=['Baja', 'Media', 'Alta'])

df_limpio = ensamble.dropna(subset=['iAUC_reportada'])
#print(df.columns)
df_modelo = pd.merge(df_limpio, pats_modelo, on = ['patient','etapas'], how='left')
#print(df_modelo)



p_train = 0.70 # Porcentaje de train.

df_modelo['is_train'] = np.random.uniform(0, 1, len(df_modelo)) <= p_train 
train = df_modelo[df_modelo['is_train']==True].copy()
test = df_modelo[df_modelo['is_train']==False].copy()
df_modelo = df_modelo.drop('is_train', axis = 1)

def procesar(df_modelo):
    
    df_modelo['age'] = pd.cut(df_modelo['age'], bins = [0,29,39,49,59,130], right=False, labels=None)  
    df_modelo.loc[df_modelo['sex'] == 'Female', 'waist_circumference'] = pd.cut(df_modelo.loc[df_modelo['sex'] == 'Female', 'waist_circumference'],
                                                                    bins=[0,80, 88, 1000], labels= ['Bajo Riesgo', 'Riesgo Elevado', 'Riesgo Muy Elevado'])
    df_modelo.loc[df_modelo['sex'] == 'Male','waist_circumference'] = pd.cut(df_modelo.loc[df_modelo['sex'] == 'Male', 'waist_circumference'],
                                                                    bins=[0,94, 102, 1000], labels= ['Bajo Riesgo', 'Riesgo Elevado', 'Riesgo Muy Elevado'])
    df_modelo['tag'] = pd.cut(df_modelo['tag'], bins = [0, 150, 200, 500, 2000], right=False, labels=['Normal', 'Limite Alto', 'Alto', 'Muy Alto'])    
    df_modelo['hdl'] = pd.cut(df_modelo['hdl'], bins = [0, 60, 1000], right=False, labels=["Bajo", "Normal"])
    df_modelo['height'] = pd.qcut(df_modelo['height'], q = 10, labels = None)
    df_modelo['pas'] = pd.cut(df_modelo['pas'], bins = [0,90,120,129,139,500], labels = ['Baja','Normal','Elevada', 'Hipertension 1', 'Hipertension 2'])
    df_modelo['pad'] = pd.cut(df_modelo['pad'], bins = [0,60,80,89,99,500], labels = ['Baja','Normal','Elevada', 'Hipertension 1', 'Hipertension 2'])
    df_modelo['pcr'] = pd.cut(df_modelo['pcr'], bins = [0,3,10,100, 500, 1000], right=False, labels=['Normal', 'Aumento Leve', 'Aumento Moderado', 'Aumento Alto', 'Aumento Muy Alto'])
    df_modelo['hba1c'] = pd.cut(df_modelo['hba1c'], bins = [0,5.7, 6.4, 100 ], right=False, labels=['Normal', 'Prediabetes', 'Diabetes'])
    df_modelo['ldl'] = pd.cut(df_modelo['ldl'], bins = [0, 100, 130, 160, 190, 1000], right=False, labels=['Optimo', 'Casi optimo', 'Limite Alto', 'Alto', 'Muy alto'])

    return df_modelo

train = procesar(train)
test = procesar(test)
train = train.drop(['patient','etapas','is_train'], axis=1)
test = test.drop(['patient','etapas','is_train'], axis=1)
print(train)

def entrenar_modelo(train, clase_interes, clase_col="iAUC_reportada"):
    
    X = train.drop(clase_col, axis=1)
    y = train[clase_col]
    
    # Probabilidades a priori
    p_c = (y == clase_interes).mean()  # P(C)
    p_no_c = 1 - p_c                   # P(¬C)
    
    # Probabilidades condicionales
    categorias = X.columns.unique()
    prob_Xi_dado_C = {}
    prob_Xi_dado_noC = {}
    
    for col in categorias:
        # Df para la clase de interés
        df_clase = train[y == clase_interes]
        prob_Xi_dado_C[col] = df_clase[col].value_counts(normalize=True).to_dict()
        
        # Df para el complemento
        df_no_clase = train[y != clase_interes]
        prob_Xi_dado_noC[col] = df_no_clase[col].value_counts(normalize=True).to_dict()
    
    # Guardar modelo en un dict
    modelo = {
        'p_c': p_c,
        'p_no_c': p_no_c,
        'prob_Xi_dado_C': prob_Xi_dado_C,
        'prob_Xi_dado_noC': prob_Xi_dado_noC,
        'categorias': categorias,
        'clase_interes': clase_interes
    }
    
    return modelo


def generar_tabla(modelo, df_datos, col_clase, clase_interes):
  
    data = modelo
    rows = []

    for var in data['prob_Xi_dado_C']:
        categorias = set(data['prob_Xi_dado_C'][var].keys()) | set(data['prob_Xi_dado_noC'][var].keys())
        
        for q in sorted(categorias):
            # Calcular conteos
            nx = (df_datos[var] == q).sum()
            nc = (df_datos[col_clase] == clase_interes).sum()
            ncx = ((df_datos[var] == q) & (df_datos[col_clase] == clase_interes)).sum()

            rows.append({
                'variable': var,
                'cat': q,
                'p_c': data['p_c'],
                'p_no_c': data['p_no_c'],
                'p_xi_C': data['prob_Xi_dado_C'][var].get(q, np.nan),
                'p_xi_noC': data['prob_Xi_dado_noC'][var].get(q, np.nan),
                'nx': nx,
                'nc': nc,
                'ncx': ncx
            })

    df = pd.DataFrame(rows)
    df = df.set_index(['variable', 'cat'])
    return df

def predecir_test(test, modelo):
    
    df_pred = test.copy()
    
    # Extraer componentes del modelo
    p_c = modelo['p_c']
    p_no_c = modelo['p_no_c']
    prob_Xi_dado_C = modelo['prob_Xi_dado_C']
    prob_Xi_dado_noC = modelo['prob_Xi_dado_noC']
    categorias = modelo['categorias']
    clase_interes = modelo['clase_interes']
    
    
    def calcular_score(fila):
        score = np.log(p_c / p_no_c)
        
        for col in categorias:
            valor = fila[col]
            
            # Obtener P(Xi|C)
            p_xi_c = prob_Xi_dado_C[col].get(valor, 1e-6)
            
            # Obtener P(Xi|¬C) 
            p_xi_noC = prob_Xi_dado_noC[col].get(valor, 1e-6)
            if p_xi_noC == 0:
                p_xi_noC = 1e-6  # Evitar divisiones por cero
            
            # Sumar al score
            score += np.log(p_xi_c / p_xi_noC)
        
        return score
    
    # Aplicar score a cada fila
    df_pred["Score_" + clase_interes] = df_pred.apply(calcular_score, axis=1)
    
    return df_pred


# Entrenar el modelo
modelo_Alta = entrenar_modelo(train, 'Alta')
modelo_Media = entrenar_modelo(train, 'Media')
modelo_Baja = entrenar_modelo(train, 'Baja')


test_Alta = predecir_test(test, modelo_Alta)
#print(test_Alta)
test_Media = predecir_test(test, modelo_Media)
test_Baja = predecir_test(test, modelo_Baja)


#################### ALTA #######################################

alta = test_Alta[['iAUC_reportada', 'Score_Alta']].copy()
alta = alta.sort_values(by = 'Score_Alta')
alta['Score_Alta'] = alta['Score_Alta'].replace(-np.inf, alta['Score_Alta'][alta['Score_Alta'] != -np.inf].min() -1)
alta['Deciles'] = pd.qcut(alta['Score_Alta'], q = 10, labels = [f'D{i}' for i in range(1,11)])


dict_y = {}
dict = {}

for decil in alta['Deciles'].unique():
    decil_fl = (alta[alta['Deciles'] == decil ])
    dict_y[decil] = len(decil_fl[decil_fl['iAUC_reportada'] == 'Alta'])
    dict[decil] = len(decil_fl) - dict_y[decil]

print('-------------ALTA----------------')
print(alta)
r = pd.DataFrame([dict_y, dict], index=["Reales", "Complemento"])
r["Total"] = r.sum(axis=1)  
print(r)  


tabla_Alta = generar_tabla(modelo_Alta, train, "iAUC_reportada", "Alta") 

############################### BAJA ####################################


baja = test_Baja[['iAUC_reportada', 'Score_Baja']].copy()
baja = baja.sort_values(by = 'Score_Baja')
baja['Score_Baja'] = baja['Score_Baja'].replace(-np.inf, baja['Score_Baja'][baja['Score_Baja'] != -np.inf].min() -1)
baja['Deciles'] = pd.qcut(baja['Score_Baja'], q = 10, labels = [f'D{i}' for i in range(1,11)])


dict_y = {}
dict = {}

for decil in baja['Deciles'].unique():
    decil_fl = (baja[baja['Deciles'] == decil ])
    dict_y[decil] = len(decil_fl[decil_fl['iAUC_reportada'] == 'Baja'])
    dict[decil] = len(decil_fl) - dict_y[decil]


print('-------------BAJA----------------')
print(baja)
r = pd.DataFrame([dict_y, dict], index=["Reales", "Complemento"])
r["Total"] = r.sum(axis=1)  
print(r)


tabla_Baja = generar_tabla(modelo_Baja, train, "iAUC_reportada", "Baja")
#print(tabla_Baja)  

############################### MEDIA ####################################


media = test_Media[['iAUC_reportada', 'Score_Media']].copy()
media = media.sort_values(by = 'Score_Media')
media['Score_Media'] = media['Score_Media'].replace(-np.inf, media['Score_Media'][media['Score_Media'] != -np.inf].min() -1)
media['Deciles'] = pd.qcut(media['Score_Media'], q = 10, labels = [f'D{i}' for i in range(1,11)])


dict_y = {}
dict = {}

for decil in media['Deciles'].unique():
    decil_fl = (media[media['Deciles'] == decil ])
    dict_y[decil] = len(decil_fl[decil_fl['iAUC_reportada'] == 'Media'])
    dict[decil] = len(decil_fl) - dict_y[decil]


print('-------------MEDIA----------------')
print(media)
r = pd.DataFrame([dict_y, dict], index=["Reales", "Complemento"])
r["Total"] = r.sum(axis=1)  
print(r) 

tabla_Media = generar_tabla(modelo_Media, train, "iAUC_reportada", "Media")
#print(tabla_Media)  


with pd.ExcelWriter("probabilidades_Mpats.xlsx") as writer:
    tabla_Alta.to_excel(writer, sheet_name="Alta")
    tabla_Media.to_excel(writer, sheet_name="Media")
    tabla_Baja.to_excel(writer, sheet_name="Baja")

  










