# ==============================================================================
# Script de análisis del dataset WESAD (Wearable Stress and Affect Detection)
# ==============================================================================
# Este script realiza un análisis completo del dataset WESAD, que contiene
# señales fisiológicas (ECG, EDA, Temperatura, Respiración, etc.) capturadas
# con sensores en el pecho de varios sujetos bajo distintas condiciones
# emocionales (estrés, diversión, baseline, meditación, etc.).
#
# El objetivo principal es identificar qué señales fisiológicas son más útiles
# para distinguir entre estados de ESTRÉS y NO ESTRÉS, usando dos técnicas
# de selección de características:
#   1. Fisher Score: mide qué tan separables son dos clases para cada variable.
#   2. Forward Selection: selecciona variables relevantes y poco redundantes.
#
# El análisis se realiza tanto de forma GLOBAL (todos los sujetos juntos)
# como POR SUJETO (cada persona por separado), y se genera un reporte PDF
# con gráficas y tablas de resultados.
# ==============================================================================

import os          # Para navegar carpetas y verificar rutas de archivos
import pickle      # Para cargar archivos .pkl (formato serializado de Python)
import numpy as np # Para operaciones numéricas (infinito, arrays, etc.)
import pandas as pd          # Para manipulación de datos tabulares (DataFrames)
import matplotlib.pyplot as plt  # Para generación de gráficas

# StandardScaler normaliza los datos: les resta la media y divide entre la
# desviación estándar, dejando cada variable con media=0 y varianza=1.
# Esto es importante para que las variables con escalas grandes no dominen
# sobre las de escalas pequeñas en el análisis.
from sklearn.preprocessing import StandardScaler

# FPDF es una librería para generar archivos PDF desde Python.
from fpdf import FPDF


# ================================
# CLASE PARA EL REPORTE PDF
# ================================
# Se crea una clase que hereda de FPDF para personalizar el encabezado.
# Al heredar de FPDF, cada página del PDF generado incluirá automáticamente
# el título "Reporte Analisis Dataset WESAD" centrado en la parte superior.

class ReporteWESAD(FPDF):

    def header(self):
        """Encabezado que aparece automáticamente en cada página del PDF."""
        self.set_font('Arial','B',16)  # Fuente Arial, negrita, tamaño 16
        self.cell(0,10,'Reporte Analisis Dataset WESAD',0,1,'C')  # Texto centrado
        self.ln(5)  # Salto de línea de 5mm después del título


# ================================
# LIMPIEZA DE OUTLIERS (VALORES ATÍPICOS)
# ================================
# Los outliers son valores extremos que no representan el comportamiento
# normal de una señal. Pueden deberse a errores del sensor, movimiento
# del sujeto, o artefactos en la medición.
#
# Estrategia: se definen rangos fisiológicamente plausibles para cada señal.
# Los valores que caen fuera de estos rangos se reemplazan por la MEDIANA
# de los valores válidos. Se usa la mediana (y no la media) porque es
# más robusta frente a valores extremos.

def limpiar_outliers(df):

    # Rangos válidos para cada señal fisiológica:
    # - Temp: temperatura corporal entre 20°C y 45°C
    # - Resp: señal de respiración entre -35 y 35
    # - ECG: electrocardiograma entre -1.5 y 1.5 mV
    # - EDA: actividad electrodérmica entre 0 y 60 µS
    filtros = {
        'Temp': (20,45),
        'Resp': (-35,35),
        'ECG': (-1.5,1.5),
        'EDA': (0,60)
    }

    for col, rango in filtros.items():

        if col in df.columns:

            lim_inf, lim_sup = rango

            # Crear una máscara booleana: True donde el valor está fuera de rango
            mask = (df[col] < lim_inf) | (df[col] > lim_sup)

            # Calcular la mediana solo con los valores que SÍ están dentro del rango
            mediana = df.loc[~mask, col].median()

            # Reemplazar los outliers con esa mediana
            df.loc[mask, col] = mediana

    return df


# ================================
# FISHER SCORE (STRESS VS NO STRESS)
# ================================
# El Fisher Score es una métrica de selección de características que mide
# qué tan bien una variable separa dos clases (en este caso: estrés vs no estrés).
#
# Fórmula:  Fisher = (μ1 - μ2)² / (σ1² + σ2²)
#   - μ1, μ2: medias de la variable en cada clase
#   - σ1², σ2²: varianzas de la variable en cada clase
#
# Interpretación:
#   - Un Fisher Score ALTO significa que las medias de las dos clases están
#     muy separadas y/o las varianzas son pequeñas → la variable es buena
#     para distinguir estrés de no estrés.
#   - Un Fisher Score BAJO significa que las clases se solapan mucho → la
#     variable no aporta información discriminativa.

def fisher_score(X, y):

    scores = {}  # Diccionario para almacenar el score de cada variable

    for col in X.columns:

        # Separar los valores de esta variable según la clase
        stress = X.loc[y == 1, col]       # Valores cuando hay estrés
        no_stress = X.loc[y == 0, col]    # Valores cuando NO hay estrés

        # Calcular media y varianza para cada grupo
        mean1 = stress.mean()
        mean2 = no_stress.mean()

        var1 = stress.var()
        var2 = no_stress.var()

        # Aplicar la fórmula de Fisher. Si ambas varianzas son 0 (variable
        # constante), se asigna un score de 0 para evitar división por cero.
        fisher = ((mean1 - mean2)**2) / (var1 + var2) if (var1 + var2) != 0 else 0

        scores[col] = fisher

    # Convertir el diccionario a DataFrame para facilitar visualización
    fisher_df = pd.DataFrame.from_dict(
        scores,
        orient='index',
        columns=['Fisher']
    )

    # Ordenar de mayor a menor: las mejores variables quedan arriba
    fisher_df = fisher_df.sort_values(by='Fisher', ascending=False)

    return fisher_df


# ================================
# FORWARD SELECTION (SELECCIÓN HACIA ADELANTE)
# ================================
# Forward Selection es un método greedy (voraz) de selección de características.
# En cada paso, elige la siguiente mejor variable considerando dos criterios:
#
#   1. Relevancia: que la variable tenga un Fisher Score alto (buena para
#      separar estrés de no estrés).
#   2. No redundancia: que la variable no esté muy correlacionada con las
#      variables ya seleccionadas (para evitar información repetida).
#
# La puntuación combinada es:
#   score = alpha1 * Fisher - alpha2 * correlación_promedio
#
# - alpha1 = 0.7 → se da más peso a la relevancia (70%)
# - alpha2 = 0.3 → se penaliza la redundancia (30%)
# - k = 5 → se seleccionan las 5 mejores variables

def forward_selection(X, fisher_df, k=5, alpha1=0.7, alpha2=0.3):

    selected = []  # Lista de variables seleccionadas

    # Convertir Fisher Scores a diccionario para acceso rápido
    fisher_dict = fisher_df['Fisher'].to_dict()

    # Paso 1: la primera variable seleccionada es la de mayor Fisher Score
    first = fisher_df.index[0]
    selected.append(first)

    # Paso 2: agregar variables una a una hasta llegar a k
    while len(selected) < k:

        best_score = -np.inf  # Inicializar con el peor score posible
        best_feature = None

        # Evaluar cada variable candidata (que no haya sido seleccionada aún)
        for feature in X.columns:

            if feature in selected:
                continue  # Saltar variables ya seleccionadas

            # Obtener su Fisher Score (relevancia)
            fisher = fisher_dict.get(feature, 0)

            # Calcular la correlación promedio con las variables ya seleccionadas
            # Una correlación alta indica redundancia (miden lo mismo)
            corr = 0
            for s in selected:
                corr += abs(X[s].corr(X[feature]))
            corr = corr / len(selected)

            # Puntuación combinada: maximizar relevancia, minimizar redundancia
            score = alpha1 * fisher - alpha2 * corr

            if score > best_score:
                best_score = score
                best_feature = feature

        # Agregar la mejor variable encontrada en esta iteración
        selected.append(best_feature)

    return selected


# ================================
# CARGAR DATASET WESAD
# ================================
# El dataset WESAD almacena los datos de cada sujeto en un archivo .pkl
# (pickle) independiente. Cada archivo contiene un diccionario con:
#   - data['signal']['chest']: señales del sensor RespiBAN (pecho)
#     que incluye ECG, EDA, EMG, Temp, ACC (acelerómetro 3 ejes), Resp
#   - data['label']: etiquetas de la condición emocional en cada instante
#     (0=no definido, 1=baseline, 2=estrés, 3=diversión, 4=meditación, etc.)
#
# Esta función carga todos los sujetos, construye un DataFrame unificado,
# elimina las muestras sin etiqueta (label=0) y limpia outliers.

def cargar_dataset() -> pd.DataFrame:

    base_path = "archive/WESAD"  # Ruta donde están las carpetas S2, S3, ...

    dataset_total = []  # Lista para acumular DataFrames de cada sujeto

    # Listar carpetas de sujetos (S2, S3, ..., S17) y ordenar numéricamente
    sujetos = [s for s in os.listdir(base_path) if s.startswith('S')]
    sujetos.sort(key=lambda x: int(x[1:]))

    for sujeto in sujetos:

        # Construir la ruta al archivo pickle del sujeto
        path = f"{base_path}/{sujeto}/{sujeto}.pkl"

        if not os.path.exists(path):
            continue  # Saltar si el archivo no existe

        print("Cargando", sujeto)

        # Cargar el archivo pickle (formato binario)
        with open(path,'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Extraer las señales del sensor de pecho (chest)
        signals = data['signal']['chest']

        df_list = []

        # Iterar por cada señal (ECG, EDA, Temp, ACC, etc.)
        for name, values in signals.items():

            # Algunas señales son multidimensionales (ej: ACC tiene 3 ejes: x,y,z)
            # En ese caso, se crea una columna separada por cada eje
            if len(values.shape) > 1 and values.shape[1] > 1:

                cols = [f"{name}_{i}" for i in range(values.shape[1])]
                df_temp = pd.DataFrame(values, columns=cols)

            else:
                # Señales unidimensionales (ECG, EDA, Temp, etc.)
                df_temp = pd.DataFrame(values, columns=[name])

            df_list.append(df_temp)

        # Unir todas las señales en un solo DataFrame (columna por señal)
        df = pd.concat(df_list, axis=1)

        # Agregar las etiquetas de condición emocional
        labels = pd.Series(data['label'], name="label")
        df['label'] = labels[:len(df)]  # Recortar por si hay diferencia de longitud

        # Agregar identificador del sujeto para análisis individual posterior
        df['Subject'] = sujeto

        # Eliminar filas con label=0 ("no definido" / transiciones entre condiciones)
        df = df[df['label'] != 0]

        # Limpiar valores atípicos en las señales fisiológicas
        df = limpiar_outliers(df)

        dataset_total.append(df)

    # Concatenar los DataFrames de todos los sujetos en uno solo
    dataset_total = pd.concat(dataset_total)

    return dataset_total


# ================================
# GRÁFICAS DE ANÁLISIS EXPLORATORIO (EDA)
# ================================
# EDA = Exploratory Data Analysis (Análisis Exploratorio de Datos)
# Es el primer paso en cualquier análisis: visualizar los datos para entender
# su distribución, detectar patrones, y encontrar posibles problemas.
#
# Se generan 3 tipos de gráficas:
#   1. Histogramas: muestran la distribución (frecuencia) de cada variable
#   2. Boxplots: muestran la mediana, cuartiles y outliers de cada variable
#   3. Matriz de correlación: muestra qué tan linealmente relacionadas
#      están las variables entre sí (valores cercanos a ±1 = alta correlación)

def graficas_eda(df):

    # Separar solo las columnas numéricas de señales (excluir label y Subject)
    features = df.drop(columns=['label','Subject'])

    # --- Histogramas ---
    # Cada gráfica muestra cuántas veces aparece cada rango de valores
    # Útil para ver si los datos siguen una distribución normal, sesgada, etc.
    features.hist(figsize=(15,10), bins=50)
    plt.tight_layout()
    plt.savefig("histogramas.png")
    plt.close()

    # --- Boxplots (diagramas de caja) ---
    # Muestran: mediana (línea central), rango intercuartil (caja),
    # y valores extremos (puntos fuera de los bigotes)
    plt.figure(figsize=(15,8))
    features.boxplot(rot=90)
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.close()

    # --- Matriz de correlación ---
    # Calcula el coeficiente de Pearson entre cada par de variables:
    #   +1 = correlación positiva perfecta
    #    0 = sin correlación lineal
    #   -1 = correlación negativa perfecta
    corr = features.corr()

    plt.figure(figsize=(10,8))
    plt.imshow(corr, cmap='coolwarm')  # Mapa de calor: rojo=positiva, azul=negativa
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Matriz de Correlacion")

    plt.tight_layout()
    plt.savefig("correlacion.png")
    plt.close()

    return corr


# ================================
# GENERACIÓN DEL REPORTE PDF
# ================================
# Esta función ensambla todas las secciones del reporte final en un PDF:
#   1. Estadísticas descriptivas (count, mean, std, min, max, etc.)
#   2. Gráficas de EDA (histogramas, boxplots, correlación)
#   3. Análisis global: Fisher Score y Forward Selection para todos los sujetos
#   4. Análisis por sujeto: Fisher Score y Forward Selection individuales

def generar_reporte(df, corr,
                    fisher_global, forward_global,
                    tabla_fisher_subject, tabla_forward_subject):

    pdf = ReporteWESAD()  # Crear el PDF con el encabezado personalizado

    # Salto de página automático cuando se llega al margen inferior (15mm)
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Sección 1: Estadísticas descriptivas ---
    # df.describe() genera: count, mean, std, min, 25%, 50%, 75%, max
    # .T transpone para que las variables queden como filas (más legible)
    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Estadisticas Descriptivas",0,1)

    desc = df.describe().round(4).T

    pdf.set_font('Courier','',9)  # Courier (monoespaciada) para tablas alineadas
    pdf.multi_cell(0,5, desc.to_string())

    # --- Sección 2: Gráficas EDA ---
    pdf.add_page()
    pdf.cell(0,10,"Histogramas",0,1)
    pdf.image("histogramas.png", x=10, w=190)

    pdf.add_page()
    pdf.cell(0,10,"Boxplots",0,1)
    pdf.image("boxplots.png", x=10, w=190)

    pdf.add_page()
    pdf.cell(0,10,"Matriz de Correlacion",0,1)
    pdf.image("correlacion.png", x=10, w=180)

    # --- Sección 3: Análisis global (todos los sujetos combinados) ---
    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Ranking Global de Caracteristicas (Fisher Score)",0,1)

    pdf.set_font('Courier','',9)
    pdf.multi_cell(0,5, fisher_global.to_string())

    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Top 5 Caracteristicas Globales (Forward Selection)",0,1)

    pdf.set_font('Courier','',10)
    pdf.multi_cell(0,5, str(forward_global))

    # --- Sección 4: Análisis por sujeto individual ---
    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Ranking Fisher por Sujeto",0,1)

    pdf.set_font('Courier','',9)
    pdf.multi_cell(0,5, tabla_fisher_subject.to_string())

    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Top 5 Caracteristicas por Sujeto (Forward Selection)",0,1)

    pdf.set_font('Courier','',10)
    pdf.multi_cell(0,5, tabla_forward_subject.to_string())

    # Guardar el PDF en disco
    pdf.output("Reporte_Final_WESAD.pdf")


# ================================
# FUNCIÓN PRINCIPAL (MAIN)
# ================================
# Orquesta todo el pipeline de análisis:
#   1. Cargar y limpiar los datos de todos los sujetos
#   2. Generar gráficas exploratorias (EDA)
#   3. Análisis global: Fisher Score + Forward Selection con todos los datos
#   4. Análisis por sujeto: lo mismo pero para cada persona individualmente
#   5. Generar el reporte PDF con todos los resultados

def main():

    # ---- Paso 1: Cargar el dataset completo ----
    print("Cargando dataset...")
    df = cargar_dataset()

    print("Dataset size:", df.shape)  # (filas, columnas) para verificar la carga

    # ---- Paso 2: Análisis exploratorio de datos (EDA) ----
    print("Generando EDA...")
    corr = graficas_eda(df)

    # =========================
    # Paso 3: ANÁLISIS GLOBAL
    # =========================
    # Se analizan TODOS los sujetos juntos para encontrar patrones generales

    # Separar variables predictoras (X) de la variable objetivo (label)
    X = df.drop(columns=['label','Subject'])

    # Crear variable binaria: 1 = estrés (label=2), 0 = cualquier otra condición
    # El dataset original usa: 1=baseline, 2=estrés, 3=diversión, 4=meditación
    y = df['label'].apply(lambda x: 1 if x == 2 else 0)

    # Normalizar los datos (media=0, varianza=1) para que todas las variables
    # sean comparables en la misma escala al calcular Fisher Score
    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),  # Ajustar y transformar en un solo paso
        columns=X.columns,        # Mantener los nombres de las columnas
        index=X.index             # Mantener los índices originales
    )

    # Calcular Fisher Score global (todas las muestras de todos los sujetos)
    fisher_global = fisher_score(X_scaled, y)

    # Seleccionar las 5 mejores variables globales con Forward Selection
    forward_global = forward_selection(X_scaled, fisher_global)

    print("\nRanking Fisher GLOBAL")
    print(fisher_global.head())

    print("\nTop 5 GLOBAL")
    print(forward_global)

    # =========================
    # Paso 4: ANÁLISIS POR SUJETO
    # =========================
    # Se repite el análisis para cada sujeto por separado.
    # Esto permite ver si las variables más importantes varían entre personas
    # (por ejemplo, un sujeto puede responder más con EDA y otro con ECG).

    subjects = df['Subject'].unique()  # Lista de sujetos únicos

    tabla_fisher = []   # Acumular resultados Fisher de cada sujeto
    tabla_forward = []  # Acumular selecciones Forward de cada sujeto

    for s in subjects:

        print("\nSUJETO", s)

        # Filtrar datos solo de este sujeto
        df_s = df[df['Subject'] == s]

        X = df_s.drop(columns=['label','Subject'])

        # Misma binarización: estrés=1, resto=0
        y = df_s['label'].apply(lambda x: 1 if x == 2 else 0)

        # Normalizar los datos de este sujeto individualmente
        scaler = StandardScaler()

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Calcular Fisher Score para este sujeto
        fisher = fisher_score(X_scaled, y)

        # Guardar cada score en la tabla acumulada
        for f,score in fisher['Fisher'].items():
            tabla_fisher.append([s,f,score])

        # Seleccionar las 5 mejores variables para este sujeto
        selected = forward_selection(X_scaled, fisher)

        for f in selected:
            tabla_forward.append([s,f])

    # Convertir las listas acumuladas a DataFrames para el reporte
    tabla_fisher = pd.DataFrame(
        tabla_fisher,
        columns=["Subject","Feature","Fisher Score"]
    )

    tabla_forward = pd.DataFrame(
        tabla_forward,
        columns=["Subject","Selected Feature"]
    )

    # ---- Paso 5: Generar el reporte PDF final ----
    print("Generando reporte PDF...")

    generar_reporte(
        df,
        corr,
        fisher_global,
        forward_global,
        tabla_fisher,
        tabla_forward
    )

    print("Reporte generado: Reporte_Final_WESAD.pdf")


# Este bloque solo se ejecuta si el script se corre directamente
# (python general_suj.py), pero NO si se importa como m\u00f3dulo desde otro script.
# Es una convenci\u00f3n est\u00e1ndar de Python para definir el punto de entrada.
if __name__ == "__main__":
    main()