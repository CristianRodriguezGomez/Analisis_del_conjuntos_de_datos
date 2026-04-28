import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===================== CONFIGURACIÓN =====================
BASE_PATH = "archive/WESAD" 
FS_CHEST = 700 
WINDOW_SEC = 120 
SUBJECTS = [f'S{i}' for i in range(2, 18) if i != 12]

# ===================== FUNCIONES DESCRIPTIVAS =====================
#el filtro elimina el "ruido" de alta frecuencia (como interferencia eléctrica)
def filtro_butterworth(datos, frecuencia_corte=1.0, fs=700):
    nyq = 0.5 * fs
    normal_cutoff = frecuencia_corte / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, datos)

def extraer_estadisticos(segmento, nombre_base):
    return {
        f"{nombre_base}_mean": np.mean(segmento),
        f"{nombre_base}_std": np.std(segmento),
        f"{nombre_base}_min": np.min(segmento),
        f"{nombre_base}_max": np.max(segmento)
    }


def calcular_fisher(df):
    features = [c for c in df.columns if c not in ['Label', 'Subject']]
    c0 = df[df['Label'] == 0]
    c1 = df[df['Label'] == 1]
    
    detalles_calculo = []
    ranking = {}
    
    for f in features:
        m0, m1 = c0[f].mean(), c1[f].mean()
        v0, v1 = c0[f].var(), c1[f].var()
        
        numerador = (m1 - m0) ** 2
        denominador = v1 + v0
        score = numerador / denominador if denominador != 0 else 0
        
        ranking[f] = score
        
        detalles_calculo.append({
            'Caracteristica': f,
            'Media_Clase_0': m0,
            'Media_Clase_1': m1,
            'Varianza_Clase_0': v0,
            'Varianza_Clase_1': v1,
            'Numerador_(m1-m0)^2': numerador,
            'Denominador_(v1+v0)': denominador,
            'Puntaje_Fisher': score
        })
        
    df_detalles = pd.DataFrame(detalles_calculo).sort_values(by='Puntaje_Fisher', ascending=False)
    return pd.Series(ranking).sort_values(ascending=False), df_detalles


def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, titulo):
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    
    # Obtener componentes de la matriz de confusión
    # tn=Verdaderos Negativos, fp=Falsos Positivos, fn=Falsos Negativos, tp=Verdaderos Positivos
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    # Cálculos según las fórmulas solicitadas
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    error_rate = (fp + fn) / total
    sensibilidad = tp / (tp + fn) if (tp + fn) != 0 else 0 # Recall
    especificidad = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f_score = (2 * precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) != 0 else 0

    print(f"\n--- MÉTRICAS DETALLADAS: {titulo} ---")
    print(f"1. Exactitud (Accuracy):      {accuracy:.4f}")
    print(f"2. Tasa de Error:             {error_rate:.4f}")
    print(f"3. Sensibilidad (Recall):     {sensibilidad:.4f}")
    print(f"4. Especificidad:             {especificidad:.4f}")
    print(f"5. Precisión:                 {precision:.4f}")
    print(f"6. F-Score:                   {f_score:.4f}")
    
    print("\nMatriz de Confusión:")
    print(f"   [TN: {tn}  FP: {fp}]")
    print(f"   [FN: {fn}  TP: {tp}]")
    
    return accuracy

# ===================== EJECUCIÓN DEL PIPELINE =====================

print("--- FASE 1: PREPROCESAMIENTO ---")
dataset_total = []

for s_id in SUBJECTS:
    ruta = os.path.join(BASE_PATH, s_id, f"{s_id}.pkl")
    if not os.path.exists(ruta): continue

    with open(ruta, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    chest, wrist, labels = data['signal']['chest'], data['signal']['wrist'], data['label']
    eda_clean = filtro_butterworth(chest['EDA'].flatten())
    muestras_win = WINDOW_SEC * FS_CHEST
    
    for i in range(0, len(labels) - muestras_win, muestras_win):
        # Sincronización y etiqueta binaria
        moda = pd.Series(labels[i : i + muestras_win]).mode()[0]
        if moda not in [1, 2, 3, 4]: continue
    

        # Clase 1: Estrés (2), Clase 0: No-Estrés (1, 3, 4)
        fila = {'Label': 1 if moda == 2 else 0, 'Subject': s_id}
        
        # --- DISPOSITIVO: PECHO (RespiBAN / RBAN) ---
        # Sensores de canal único
        fila.update(extraer_estadisticos(chest['ECG'][i:i+muestras_win], "RBAN_ECG"))
        fila.update(extraer_estadisticos(eda_clean[i:i+muestras_win], "RBAN_EDA"))
        fila.update(extraer_estadisticos(chest['Temp'][i:i+muestras_win], "RBAN_Temp"))
        fila.update(extraer_estadisticos(chest['EMG'][i:i+muestras_win], "RBAN_EMG"))    
        fila.update(extraer_estadisticos(chest['Resp'][i:i+muestras_win], "RBAN_Resp"))  

        # Acelerómetro Pecho (ACC triaxial)
        acc_chest = chest['ACC'][i:i+muestras_win]
        mag_acc_chest = np.sqrt(np.sum(np.square(acc_chest), axis=1))
        fila.update(extraer_estadisticos(mag_acc_chest, "RBAN_ACC"))

        # --- DISPOSITIVO: MUÑECA (Empatica E4) ---
        # Sensores de canal único
        fila.update(extraer_estadisticos(wrist['BVP'].flatten(), "E4_BVP"))
        fila.update(extraer_estadisticos(wrist['EDA'].flatten(), "E4_EDA"))
        fila.update(extraer_estadisticos(wrist['TEMP'].flatten(), "E4_TEMP"))

        # Acelerómetro Muñeca (ACC triaxial)
        acc_wrist = wrist['ACC']
        mag_acc_wrist = np.sqrt(np.sum(np.square(acc_wrist), axis=1))
        fila.update(extraer_estadisticos(mag_acc_wrist, "E4_ACC"))

        
        dataset_total.append(fila)
    print(f"Sujeto {s_id} procesado.")


df = pd.DataFrame(dataset_total).fillna(0)

# Normalización Global
columnas_x = [c for c in df.columns if c not in ['Label', 'Subject']]
scaler = StandardScaler()
df[columnas_x] = scaler.fit_transform(df[columnas_x])

print("\n--- DIMENSIONES DE LA MATRIZ ---")
print(f"Total de instancias (filas): {df.shape[0]}")
print(f"Total de características (columnas): {len(columnas_x)}")


# ===============================================================================

# División 70-30
X_total = df[columnas_x]
y_total = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3, random_state=42)

# ===================== EXPORTACIÓN A EXCEL =====================
print("\n--- GENERANDO ARCHIVO EXCEL ---")
# Obtenemos de nuevo el ranking y los detalles para exportar
_, df_pasos_fisher = calcular_fisher(df)

with pd.ExcelWriter('Verificacion_WESAD_Fisher.xlsx') as writer:
    # Hoja 1: El dataset completo
    df.to_excel(writer, sheet_name='Dataset_Completo', index=False)
    
    # Hoja 2: Los cálculos detallados de Fisher
    df_pasos_fisher.to_excel(writer, sheet_name='Calculos_Fisher', index=False)

print("Archivo 'Verificacion_WESAD_Fisher.xlsx' creado exitosamente.")


# ... (continúa el código con FASE 2: SELECCIÓN POR FISHER)
print("\n--- FASE 2: SELECCIÓN POR FISHER ---")
ranking, _ = calcular_fisher(df)

# Obtenemos los top 5 como un objeto de Series (que sí tiene .items())
top_5_series = ranking.head(5)

# Creamos la lista de nombres para los clasificadores más adelante
top_5 = top_5_series.index.tolist()

print("Top 5 Características seleccionadas:")
for i, (feat, score) in enumerate(top_5_series.items(), 1):
    print(f"{i}. {feat:18} | Puntaje Fisher: {score:.4f}")

# ===================== FASE 3: CLASIFICACIÓN =====================

# 1. Árbol de Decisión (Todas)
evaluar_modelo(DecisionTreeClassifier(), X_train, X_test, y_train, y_test, "Árbol (Todas)")

# 2. Árbol de Decisión (Top 5 Fisher)
evaluar_modelo(DecisionTreeClassifier(), X_train[top_5], X_test[top_5], y_train, y_test, "Árbol (Top 5)")

# 3. Bayes Ingenuo (Todas)
evaluar_modelo(GaussianNB(), X_train, X_test, y_train, y_test, "Bayes (Todas)")

# 4. Bayes Ingenuo (Top 5 Fisher)
evaluar_modelo(GaussianNB(), X_train[top_5], X_test[top_5], y_train, y_test, "Bayes (Top 5)")


# ===================== FASE 4: GRÁFICAS =====================

nombres = ["Árbol (Todas)", "Árbol (Top 5)", "Bayes (Todas)", "Bayes (Top 5)"]
modelos = [
    (DecisionTreeClassifier(), X_train, X_test),
    (DecisionTreeClassifier(), X_train[top_5], X_test[top_5]),
    (GaussianNB(), X_train, X_test),
    (GaussianNB(), X_train[top_5], X_test[top_5])
]

# accuracy
accuracies = []
for modelo, xt, xv in modelos:
    modelo.fit(xt, y_train)
    accuracies.append(accuracy_score(y_test, modelo.predict(xv)))

# 2. Gráfica de Ranking de Fisher
plt.figure(figsize=(10, 6))
sns.barplot(x=ranking.head(15).values, y=ranking.head(15).index, hue=ranking.head(15).index, palette='viridis', legend=False)
plt.title('Top 15 Características - Factor de Fisher', fontsize=14)
plt.xlabel('Puntaje Fisher', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Gráfica de Comparativa de Modelos
plt.figure(figsize=(10, 6))
df_res = pd.DataFrame({'Modelo': nombres, 'Exactitud': accuracies})
ax = sns.barplot(x='Modelo', y='Exactitud', data=df_res, hue='Modelo', palette='magma', legend=False)

# Etiquetas automáticas sobre las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height()*100:.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), 
                textcoords='offset points', fontsize=11, fontweight='bold')

plt.title('Exactitud', fontsize=14)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
