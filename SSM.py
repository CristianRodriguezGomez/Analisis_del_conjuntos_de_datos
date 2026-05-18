import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB

# 1. CARGA DE DATOS Y CONFIGURACIÓN INICIAL
print("=== ETAPA A) CARGA DE DATOS ===")
# Cargamos el archivo original (asegúrate de que esté en la misma carpeta)
df = pd.read_csv("StressLevelDataset.csv")

# Validación física reportada en el EDA
print(f"Total de registros: {df.shape[0]} | Columnas: {df.shape[1]}")
print(f"Valores faltantes detectados: {df.isnull().sum().sum()}")
print(f"Filas duplicadas: {df.duplicated().sum()}\n")

# Definición del Espacio de Estados
# Variable Objetivo especificada por la profesora: anxiety_level
X = df.drop(columns=['anxiety_level'])
y = df['anxiety_level']

# DIVISIÓN DE DATOS (70% Train, 30% Test) ANTES DEL PROCESO
# Se aplica stratify=y para asegurar proporciones homogéneas de la clase en las particiones
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Conjunto de Entrenamiento (Train): {X_train.shape[0]} instancias")
print(f"Conjunto de Prueba (Test): {X_test.shape[0]} instancias\n")

#PREPROCESAMIENTO Y DISCRETIZACIÓN
print("=== ETAPA B) PREPROCESAMIENTO (DISCRETIZACIÓN T=3) ===")

X_train_disc = X_train.copy()
X_test_disc = X_test.copy()

# Rutina estricta para aprender los intervalos en Train y aplicarlos en Test (Evita Data Leakage)
for col in X_train.columns:
    # Determinamos cortes balanceados en 3 intervalos basándonos únicamente en Train
    # Variables críticas numéricas amplias y métricas se unifican a rango ordinal [1, 2, 3]
    _, bins_cortes = pd.cut(X_train[col], bins=3, retbins=True, labels=[1, 2, 3])
    
    # Aseguramos los límites exteriores para evitar desbordamientos con datos de prueba
    bins_cortes[0] = -np.inf
    bins_cortes[-1] = np.inf
    
    # Transformamos Train y Test con la misma máscara matemática
    X_train_disc[col] = pd.cut(X_train[col], bins=bins_cortes, labels=[1, 2, 3]).astype(int)
    X_test_disc[col] = pd.cut(X_test[col], bins=bins_cortes, labels=[1, 2, 3]).astype(int)

# Discretizamos también la variable objetivo (Target: Nivel de Ansiedad)
_, bins_y = pd.cut(y_train, bins=3, retbins=True, labels=[0, 1, 2])
bins_y[0] = -np.inf
bins_y[-1] = np.inf

y_train_disc = pd.cut(y_train, bins=bins_y, labels=[0, 1, 2]).astype(int)
y_test_disc = pd.cut(y_test, bins=bins_y, labels=[0, 1, 2]).astype(int)

print("Distribución resultante del Target (Anxiety Level) en Entrenamiento:")
print(y_train_disc.value_counts().sort_index())
print("¡Preprocesamiento completado de forma segura sin fuga de datos!\n")

#  SELECCIÓN DE CARACTERÍSTICAS
print("=== ETAPA C) SELECCIÓN DE CARACTERÍSTICAS POR INFORMACIÓN MUTUA ===")

def calcular_informacion_mutua_discreta(X_col, y_target):
    """
    Calcula de forma exacta la Información Mutua clásica de Shannon
    para variables discretas indexadas.
    """
    # Tabla de contingencia 
    tabla_conjunta = pd.crosstab(X_col, y_target, normalize=True).values
    # Probabilidades marginales
    prob_x = pd.crosstab(X_col, y_target, normalize=True).sum(axis=1).values
    prob_y = pd.crosstab(X_col, y_target, normalize=True).sum(axis=0).values
    
    mi_score = 0.0
    for i in range(len(prob_x)):
        for j in range(len(prob_y)):
            p_xy = tabla_conjunta[i, j]
            if p_xy > 0:  # Evitamos indeterminación logarítmica log(0)
                mi_score += p_xy * np.log2(p_xy / (prob_x[i] * prob_y[j]))
    return mi_score

# Calculamos el score informacional para cada variable predictora
scores_mi = {}
for col in X_train_disc.columns:
    scores_mi[col] = calcular_informacion_mutua_discreta(X_train_disc[col], y_train_disc)

# Ordenamos el ranking de mayor a menor impacto discriminante
ranking_mi = pd.Series(scores_mi).sort_values(ascending=False)

print("Ranking Completo de Información Mutua (Target: Ansiedad):")
for var, valor in ranking_mi.items():
    print(f" - {var.ljust(28)}: {valor:.4f}")

# Extraemos el Top 5 idéntico al consolidado por tu equipo
top_5_features = list(ranking_mi.head(5).index)
print(f"\nTop 5 Características Seleccionadas: {top_5_features}\n")

# Creamos los subsets de datos con las características óptimas seleccionadas
X_train_top5 = X_train_disc[top_5_features]
X_test_top5 = X_test_disc[top_5_features]

#  APRENDIZAJE Y EVALUACIÓN
print("=== ETAPA APRENDIZAJE Y EVALUACIÓN DE MODELOS ===\n")

def evaluar_modelo(modelo, X_tr, X_te, y_tr, y_te, nombre_experimento):
    # Ajuste/Entrenamiento en el 70%
    modelo.fit(X_tr, y_tr)
    # Predicción en el 30% independiente de prueba
    preds = modelo.predict(X_te)
    
    # Cálculo formal de métricas multiclase ponderadas (weighted)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average='weighted')
    rec = recall_score(y_te, preds, average='weighted')
    f1 = f1_score(y_te, preds, average='weighted')
    cm = confusion_matrix(y_te, preds)
    
    print(f"--- RESULTADOS: {nombre_experimento} ---")
    print(f"Accuracy                 : {acc:.4f}")
    print(f"Precision (weighted)     : {prec:.4f}")
    print(f"Recall (weighted)        : {rec:.4f}")
    print(f"F1-Score (weighted)      : {f1:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print("-" * 50 + "\n")
    return acc, prec, rec, f1, cm

# EXPERIMENTO 1: Bayes Ingenuo (Todas las Variables) 
# Usamos CategoricalNB dado que el espacio de características ha sido discretizado a factores ordinales
evaluar_modelo(
    CategoricalNB(), X_train_disc, X_test_disc, y_train_disc, y_test_disc, 
    "Bayes Ingenuo (Todas las Variables)"
)

#  EXPERIMENTO 2: Bayes Ingenuo (Top 5 Características)
evaluar_modelo(
    CategoricalNB(), X_train_top5, X_test_top5, y_train_disc, y_test_disc, 
    "Bayes Ingenuo (Top 5 Información Mutua)"
)

#  EXPERIMENTO 3: Árbol de Decisión (Todas las Variables) 
# Se fija random_state para reproducibilidad matemática exacta de las ramas
evaluar_modelo(
    DecisionTreeClassifier(criterion='entropy', random_state=42), 
    X_train_disc, X_test_disc, y_train_disc, y_test_disc, 
    "Árbol de Decisión (Todas las Variables)"
)

#  EXPERIMENTO 4: Árbol de Decisión (Top 5 Características) 
evaluar_modelo(
    DecisionTreeClassifier(criterion='entropy', random_state=42), 
    X_train_top5, X_test_top5, y_train_disc, y_test_disc, 
    "Árbol de Decisión (Top 5 Información Mutua)"
)