import pandas as pd
import numpy as np
import math
import requests

np.random.seed(42)




url_api = 'https://nutricion.c3.unam.mx/nd'

df1 = pd.read_csv('test.csv')   #dataset con los datos de todas las comidas consideradas

# diferencia entre auc( pospandrial - basal) reportada
df1['reportada_PPandrial_Basal_diferencia'] = df1['reportada_PPandrial_auc'] - (df1['reportada_basal_auc']) * 2   

# df en el que solo estan los valores positivos de la diferencia
df2 = df1[df1['reportada_PPandrial_Basal_diferencia'] >= 0].copy().reset_index(drop=True)    

# crear nueva columna rangos donde categorice los valores de las diferencias positivas
df2['rangos'] = pd.cut(df2['reportada_PPandrial_Basal_diferencia'], 20, right=True, labels=None)  

bins_rangos = [-13, 2036.979, 4073.958, 13579.861]   # los rangos que se consideraron para separar las comidas
df2['iAUC_reportada'] = pd.cut(df2['reportada_PPandrial_Basal_diferencia'], bins = bins_rangos, right=True, labels=['Baja', 'Media', 'Alta'])
df_modelo = df2[['hc_total', 'fiber_total', 'lipids_total', 'protein_total', 'patient', 'etapas', 'iAUC_reportada']].copy()
df_modelo = df_modelo.dropna(subset=['iAUC_reportada'])

# Agregar columna con porcentaje de fibra y porcentaje de carbohidratos
df_modelo['total_gr'] = df_modelo[['hc_total', 'fiber_total', 'lipids_total', 'protein_total']].sum(axis=1)
df_modelo['Proporcion_hc'] = (df_modelo['hc_total'] * 100) / df_modelo['total_gr']
df_modelo['Proporcion_fiber'] = (df_modelo['fiber_total'] * 100) / df_modelo['total_gr'] #proporcion de fibra con respecto al total de lo que se consume
#proporcion de fibra y hc juntos con respecto al total de que se consume
df_modelo['total_fib_carb'] = df_modelo['fiber_total'] + df_modelo['hc_total']
df_modelo['porc_fibra_carb'] = (df_modelo['total_fib_carb'] * 100) / df_modelo['total_gr']  

print(df_modelo)
#print(df_modelo.columns)

df_modelo = df_modelo.loc[df_modelo["total_gr"] > 0.0]
df_modelo.to_csv('comidas_modelo.csv', index=False)
#print(df_modelo)


p_train = 0.70 # Porcentaje de train.

df_modelo['is_train'] = np.random.uniform(0, 1, len(df_modelo)) <= p_train 
train = df_modelo[df_modelo['is_train']==True].copy()
test = df_modelo[df_modelo['is_train']==False].copy()
df_modelo = df_modelo.drop('is_train', axis = 1)

def procesar(df_modelo):
    
    nutrientes = ['hc_total', 'fiber_total', 'lipids_total', 'protein_total', 'Proporcion_hc', 'Proporcion_fiber', 'porc_fibra_carb']    
    for nutriente in nutrientes:
        df_modelo[nutriente] = pd.qcut(
        df_modelo[nutriente], 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )
    #print(df_modelo)
    return df_modelo

train = procesar(train)
test = procesar(test)
train = train.drop(['patient','etapas','is_train','total_gr', 'total_fib_carb'], axis=1)
test = test.drop(['patient','etapas','is_train','total_gr', 'total_fib_carb'], axis=1)
print(train)
print(test)


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
#print(tabla_Alta)  

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
#print(tabla_Alta)  

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
  
with pd.ExcelWriter("probabilidades_Mcomidas.xlsx") as writer:
    tabla_Alta.to_excel(writer, sheet_name="Alta")
    tabla_Media.to_excel(writer, sheet_name="Media")
    tabla_Baja.to_excel(writer, sheet_name="Baja")



