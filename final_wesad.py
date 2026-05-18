# ===================== IMPORTS =====================
import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
DATA_PATH = "archive/WESAD"
FS = 700

# 15 Sujetos en total
ALL_SUBJECTS = [f"S{i}" for i in range(2, 12)] + [f"S{i}" for i in range(13, 18)]

# División fija propuesta: 11 Sujetos para Entrenamiento (73%) y 4 para Pruebas (27%)
TRAIN_SUBJECTS = ALL_SUBJECTS[:11]  # S2 a S11 y S13
TEST_SUBJECTS = ALL_SUBJECTS[11:]   # S14 a S17

WINDOW_SEC = 120
STEP_SEC = 60

# ===================== LOAD SUBJECT =====================
def load_subject(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

# ===================== EDA FEATURES =====================
def tonic_phasic(eda, fs):
    eda = eda[~np.isnan(eda)]
    b, a = butter(4, 0.05 / (fs / 2), btype="low")
    tonic = filtfilt(b, a, eda)
    phasic = eda - tonic
    return tonic, phasic

def extract_eda_features(eda, fs):
    tonic, phasic = tonic_phasic(eda, fs)
    peaks, _ = find_peaks(phasic, height=0.01, distance=fs)
    return len(peaks), np.trapezoid(np.abs(phasic)) / fs, np.mean(tonic)

def build_eda_table(subject):
    eda = subject["signal"]["chest"]["EDA"]
    labels = subject["label"]

    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    rows = []

    for i in range(0, len(eda) - WINDOW, STEP):
        win_eda = eda[i:i + WINDOW]
        win_labels = labels[i:i + WINDOW]

        if np.isnan(win_eda).any():
            continue

        majority = np.bincount(win_labels).argmax()
        if majority not in [1, 2, 3, 4]:
            continue

        label_bin = 1 if majority == 2 else 0
        scr, auc, tonic = extract_eda_features(win_eda, FS)

        rows.append({
            "Time": i / FS,
            "Label": label_bin,
            "EDA_SCR_count": scr,
            "EDA_Phasic_AUC": auc,
            "EDA_Tonic_Mean": tonic
        })
    return pd.DataFrame(rows)

# ===================== HRV FEATURES =====================
def build_hrv_table(subject):
    ecg = subject["signal"]["chest"]["ECG"]
    labels = subject["label"]

    cleaned = nk.ecg_clean(ecg, sampling_rate=FS)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=FS)
    rpeaks_idx = rpeaks["ECG_R_Peaks"]

    WINDOW = FS * WINDOW_SEC
    STEP = FS * STEP_SEC
    rows = []

    for start in range(0, len(ecg) - WINDOW, STEP):
        end = start + WINDOW
        peaks = rpeaks_idx[(rpeaks_idx >= start) & (rpeaks_idx < end)] - start
        if len(peaks) <= 2:
            continue

        rr_ms = np.diff(peaks) / FS * 1000
        if len(rr_ms) < 2:
            continue

        try:
            rpeaks_clean = nk.intervals_to_peaks(rr_ms, sampling_rate=FS)
            hrv_t = nk.hrv_time(rpeaks_clean, sampling_rate=FS, show=False)
            hrv_f = nk.hrv_frequency(rpeaks_clean, sampling_rate=FS, show=False)

            label_bin = np.bincount((labels[start:end] == 2).astype(int)).argmax()

            rows.append({
                "Time": (start + end) / 2 / FS,
                "Label": label_bin,
                "HRV_RMSSD": hrv_t["HRV_RMSSD"].values[0],
                "HRV_SDNN": hrv_t["HRV_SDNN"].values[0],
                "HRV_MeanNN": hrv_t["HRV_MeanNN"].values[0],
                "HRV_LF": hrv_f["HRV_LF"].values[0],
                "HRV_HF": hrv_f["HRV_HF"].values[0],
                "HRV_LFHF": hrv_f["HRV_LFHF"].values[0],
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

# ===================== DATA EXTRACTION CYCLE =====================
eda_all, hrv_all = [], []

for sid in ALL_SUBJECTS:
    print(f"Procesando Sujeto: {sid}...")
    subject = load_subject(f"{DATA_PATH}/{sid}/{sid}.pkl")

    eda_df = build_eda_table(subject)
    hrv_df = build_hrv_table(subject)

    eda_df["Subject"] = sid
    hrv_df["Subject"] = sid

    eda_all.append(eda_df)
    hrv_all.append(hrv_df)

EDA_FEATURES = pd.concat(eda_all, ignore_index=True)
HRV_FEATURES = pd.concat(hrv_all, ignore_index=True)

merged_df = pd.merge(
    EDA_FEATURES,
    HRV_FEATURES,
    on=["Time", "Label", "Subject"],
    how="inner"
)

# ===================== DIVISION POR SUJETOS (TRAIN / TEST) =====================
train_df = merged_df[merged_df["Subject"].isin(TRAIN_SUBJECTS)]
test_df = merged_df[merged_df["Subject"].isin(TEST_SUBJECTS)]

# print("\n===== DATOS DE ESPECIFICACIÓN PARA LA DIAPOSITIVA =====")
# print(f"Sujetos Entrenamiento ({len(TRAIN_SUBJECTS)}): {TRAIN_SUBJECTS}")
# print(f"Registros Entrenamiento (X_train): {len(train_df)}")
# print(f"Distribución de Clases en Train:\n{train_df['Label'].value_counts()}")

# print(f"\nSujetos Prueba ({len(TEST_SUBJECTS)}): {TEST_SUBJECTS}")
# print(f"Registros Prueba (X_test): {len(test_df)}")
# print(f"Distribución de Clases en Test:\n{test_df['Label'].value_counts()}")

# ===================== CONTEO Y REPORTE DE VENTANAS (AGREGA ESTO AQUÍ) =====================
print("\n" + "="*60)
print("     DATOS EXACTOS PARA LAS DIAPOSITIVAS (PUNTO D)     ")
print("="*60)
print(f"1. ESTRUCTURA GLOBAL DEL DATASET PROCESADO:")
print(f"   - Cantidad total de ventanas (filas):  {len(merged_df)}")
print(f"   - Número de características (columnas): {len(merged_df.columns) - 3}") # Restamos Time, Label y Subject
print(f"   - Distribución global de clases:")
print(f"     * Clase 0 (No-Estrés): {len(merged_df[merged_df['Label'] == 0])} ventanas")
print(f"     * Clase 1 (Estrés):    {len(merged_df[merged_df['Label'] == 1])} ventanas")

print("\n2. CONJUNTO DE ENTRENAMIENTO (TRAIN) - 11 SUJETOS:")
print(f"   - Sujetos asignados:           {TRAIN_SUBJECTS}")
print(f"   - Total de registros (filas):  {len(train_df)}")
print(f"   - Distribución de clases en Train:")
print(f"     * Clase 0 (No-Estrés): {len(train_df[train_df['Label'] == 0])} ventanas")
print(f"     * Clase 1 (Estrés):    {len(train_df[train_df['Label'] == 1])} ventanas")

print("\n3. CONJUNTO DE PRUEBAS (TEST) - 4 SUJETOS:")
print(f"   - Sujetos asignados:           {TEST_SUBJECTS}")
print(f"   - Total de registros (filas):  {len(test_df)}")
print(f"   - Distribución de clases en Test:")
print(f"     * Clase 0 (No-Estrés): {len(test_df[test_df['Label'] == 0])} ventanas")
print(f"     * Clase 1 (Estrés):    {len(test_df[test_df['Label'] == 1])} ventanas")
print("="*60 + "\n")





# Separar características y etiquetas
FEATURE_COLS = ["EDA_SCR_count", "EDA_Phasic_AUC", "EDA_Tonic_Mean", 
                "HRV_RMSSD", "HRV_SDNN", "HRV_MeanNN", "HRV_LF", "HRV_HF", "HRV_LFHF"]

X_train = train_df[FEATURE_COLS]
y_train = train_df["Label"].values

X_test = test_df[FEATURE_COLS]
y_test = test_df["Label"].values

# ===================== PREPROCESAMIENTO (ESCALADO Z-SCORE) =====================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS)

# ===================== SELECCIÓN DE CARACTERÍSTICAS (FACTOR DE FISHER) =====================
def fisher_factor(X, y):
    scores = {}
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    
    for col in X.columns:
        mean_0, std_0 = X_0[col].mean(), X_0[col].std()
        mean_1, std_1 = X_1[col].mean(), X_1[col].std()
        
        # Evitar división por cero
        denominator = (std_0**2 + std_1**2) if (std_0**2 + std_1**2) != 0 else 1e-6
        f_score = (mean_1 - mean_0)**2 / denominator
        scores[col] = f_score
        
    return pd.Series(scores).sort_values(ascending=False)

print("\n===== RESULTADOS DEL FACTOR DE FISHER =====")
fisher_scores = fisher_factor(X_train_scaled, y_train)
print(fisher_scores)

TOP_5_FEATURES = fisher_scores.head(5).index.tolist()
print(f"\nTop 5 Características Seleccionadas: {TOP_5_FEATURES}")


print("\n===== RESULTADOS DEL FACTOR DE FISHER =====")
fisher_scores = fisher_factor(X_train_scaled, y_train)
print(fisher_scores)

TOP_5_FEATURES = fisher_scores.head(5).index.tolist()
print(f"\nTop 5 Características Seleccionadas: {TOP_5_FEATURES}")

# ===================== AGREGAR DESDE AQUÍ PARA LA GRÁFICA =====================
plt.figure(figsize=(10, 5))
# Creamos la gráfica de barras horizontal ordenando los scores de menor a mayor para el plot
sns.barplot(x=fisher_scores.values, y=fisher_scores.index, palette="viridis")

plt.title("Ranking de Características mediante Factor de Fisher (Conjunto Train)", fontsize=14, fontweight='bold')
plt.xlabel("Score del Factor de Fisher", fontsize=12)
plt.ylabel("Características", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ajuste estético para que no se corten las etiquetas de los nombres
plt.tight_layout()
plt.show()
# ===================== TERMINA BLOQUE DE LA GRÁFICA =====================
# ===================== FUNCIÓN DE EVALUACIÓN OPTIMIZADA =====================
def run_experiment(model, X_tr, y_tr, X_te, y_te, model_name, feat_type):
    # Entrenar y predecir
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    # Obtener la matriz de confusión y desglosar cuadrantes
    cm = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular las métricas detalladas con protección de división por cero
    exactitud = (tp + tn) / (tp + tn + fp + fn)
    tasa_error = 1 - exactitud
    sensibilidad = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    f_score = (2 * precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) != 0 else 0.0
    
    # Despliegue en consola con el formato exacto de las diapositivas
    print(f"\n==================================================")
    print(f" RESULTADOS: {model_name.upper()} ({feat_type.upper()})")
    print(f"==================================================")
    print(f"MATRIZ DE CONFUSIÓN:")
    print(f"[TN: {tn:<4} FP: {fp}]")
    print(f"[FN: {fn:<4} TP: {tp}]")
    print(f"--------------------------------------------------")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Tasa de Error: {tasa_error:.4f}")
    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"F-Score: {f_score:.4f}")
    print(f"==================================================\n")
    
    # Graficar la matriz de confusión de forma estética
    plt.figure(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No-Stress", "Stress"], 
                yticklabels=["No-Stress", "Stress"],
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"{model_name}\n({feat_type})", fontsize=11, fontweight='bold')
    plt.ylabel("Clase Real (Ground Truth)", fontsize=10)
    plt.xlabel("Clase Predicha por el Modelo", fontsize=10)
    plt.tight_layout()
    plt.show()

# ===================== APRENDIZAJE Y EXPERIMENTOS =====================
print("\n===== INICIANDO EVALUACIÓN DE MODELOS EN EL SET DE TEST =====")

# Inicializar los dos clasificadores requeridos
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, class_weight="balanced")

# ----------------- EXPERIMENTO 1: TODAS LAS VARIABLES (9) -----------------
run_experiment(nb_model, X_train_scaled, y_train, X_test_scaled, y_test, 
               "Gaussian Naive Bayes", "Todas las variables")

run_experiment(dt_model, X_train_scaled, y_train, X_test_scaled, y_test, 
               "Árbol de Decisión", "Todas las variables")

# ----------------- EXPERIMENTO 2: TOP 5 DE FISHER -----------------
run_experiment(nb_model, X_train_scaled[TOP_5_FEATURES], y_train, X_test_scaled[TOP_5_FEATURES], y_test, 
               "Gaussian Naive Bayes", "Top 5 Fisher")

run_experiment(dt_model, X_train_scaled[TOP_5_FEATURES], y_train, X_test_scaled[TOP_5_FEATURES], y_test, 
               "Árbol de Decisión", "Top 5 Fisher")
