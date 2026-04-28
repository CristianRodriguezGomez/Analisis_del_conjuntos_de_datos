import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

#----------INICIO DEL PREPROCESADO, DISCRETIZACION Y RANKEO---------

# 1. CARGA Y EXPLORACIÓN INICIAL

print("=" * 60)
print("  1. CARGA Y EXPLORACIÓN DEL DATASET")
print("=" * 60)

df = pd.read_csv("StressLevelDataset.csv")

print(f"\nShape: {df.shape}")
print(f"\nColumnas ({len(df.columns)}):\n  {list(df.columns)}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")
print(f"\nDistribución de stress_level:\n{df['stress_level'].value_counts().sort_index()}")

# 2. PREPROCESAMIENTO

print("\n" + "=" * 60)
print("  2. PREPROCESAMIENTO")
print("=" * 60)

# 2.1 Sin valores nulos, pero verificamos duplicados
n_dup = df.duplicated().sum()
print(f"\nDuplicados encontrados: {n_dup}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape tras eliminar duplicados: {df.shape}")

# 2.2 Separar características y targets
FEATURES = [c for c in df.columns if c != "stress_level"]
TARGET_STRESS   = "stress_level"       # 0, 1, 2
TARGET_ANXIETY  = "anxiety_level"      # continuo 0-21
TARGET_DEPRESSION = "depression"       # continuo 0-27

print(f"\nCaracterísticas: {FEATURES}")
print(f"Target estrés (clases): {sorted(df[TARGET_STRESS].unique())}")
print(f"Rango anxiety_level: {df[TARGET_ANXIETY].min()} – {df[TARGET_ANXIETY].max()}")
print(f"Rango depression: {df[TARGET_DEPRESSION].min()} – {df[TARGET_DEPRESSION].max()}")
print(f"Rango self_esteem: {df['self_esteem'].min()} – {df['self_esteem'].max()}")

# 3. DISCRETIZACIÓN EN 3 CLASES (anxiety_level, self_esteem, depression)

print("\n" + "=" * 60)
print("  3. DISCRETIZACIÓN EN 3 CLASES")
print("=" * 60)

df_disc = df.copy()

COLS_TO_DISC = ["anxiety_level", "self_esteem", "depression"]
disc_labels  = {
    "anxiety_level": ["bajo", "medio", "alto"],
    "self_esteem":   ["bajo", "medio", "alto"],
    "depression":    ["bajo", "medio", "alto"],
}

# Usamos KBinsDiscretizer con estrategia cuantil para clases balanceadas
kbd = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
disc_values = kbd.fit_transform(df_disc[COLS_TO_DISC])

for i, col in enumerate(COLS_TO_DISC):
    disc_col = f"{col}_disc"
    df_disc[disc_col] = disc_values[:, i].astype(int)
    edges = kbd.bin_edges_[i]
    print(f"\n{col}:")
    print(f"  Bordes de bins: {np.round(edges, 2)}")
    print(f"  Distribución discreta:\n{pd.Series(df_disc[disc_col]).value_counts().sort_index()}")

# Mapear números a etiquetas legibles
for col in COLS_TO_DISC:
    disc_col = f"{col}_disc"
    df_disc[f"{col}_label"] = df_disc[disc_col].map({0: "bajo", 1: "medio", 2: "alto"})

print("\nEjemplo de discretización (5 filas):")
print(df_disc[["anxiety_level", "anxiety_level_disc",
               "self_esteem", "self_esteem_disc",
               "depression", "depression_disc"]].head())

# 4. RANKING POR INFORMACIÓN MUTUA PARA CLASIFICAR DEPRESSION

print("\n" + "=" * 60)
print("  4. RANKING POR INFORMACIÓN MUTUA → TARGET: depression_disc")
print("=" * 60)

# Características para el ranking: todo excepto las 3 discretizadas y el target original
excl = {"anxiety_level", "self_esteem", "depression",
        "anxiety_level_disc", "self_esteem_disc", "depression_disc",
        "anxiety_level_label", "self_esteem_label", "depression_label",
        "stress_level"}

feature_cols_rank = [c for c in df_disc.columns if c not in excl]
X_rank = df_disc[feature_cols_rank]
y_rank = df_disc["depression_disc"]

mi_scores = mutual_info_classif(X_rank, y_rank, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_cols_rank).sort_values(ascending=False)

print("\nRanking completo (Información Mutua respecto a depression_disc):")
print("-" * 45)
for rank, (feat, score) in enumerate(mi_series.items(), 1):
    print(f"  {rank:2d}. {feat:<35s} {score:.4f}")

# Top-5 para usar en modelos
TOP_K = 5
TOP_FEATURES_DEPRESSION = mi_series.head(TOP_K).index.tolist()
print(f"\nTop-{TOP_K} características seleccionadas: {TOP_FEATURES_DEPRESSION}")

# Visualización del ranking
plt.figure(figsize=(10, 6))
mi_series.plot(kind="barh", color="steelblue", edgecolor="black")
plt.xlabel("Información Mutua")
plt.title("Ranking de características por Información Mutua\n(Target: nivel de depresión)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("ranking_mutual_info_depression.png", dpi=150)
plt.close()
print("\nGráfica guardada: ranking_mutual_info_depression.png")

# 5. RANKING POR INFORMACIÓN MUTUA PARA CLASIFICAR ANXIETY

print("\n" + "=" * 60)
print("  5. RANKING POR INFORMACIÓN MUTUA → TARGET: anxiety_level_disc")
print("=" * 60)

y_rank_anx = df_disc["anxiety_level_disc"]
mi_scores_anx = mutual_info_classif(X_rank, y_rank_anx, random_state=42)
mi_series_anx = pd.Series(mi_scores_anx, index=feature_cols_rank).sort_values(ascending=False)

print("\nRanking completo (Información Mutua respecto a anxiety_level_disc):")
print("-" * 45)
for rank, (feat, score) in enumerate(mi_series_anx.items(), 1):
    print(f"  {rank:2d}. {feat:<35s} {score:.4f}")

TOP_FEATURES_ANXIETY = mi_series_anx.head(TOP_K).index.tolist()
print(f"\nTop-{TOP_K} características seleccionadas: {TOP_FEATURES_ANXIETY}")

plt.figure(figsize=(10, 6))
mi_series_anx.plot(kind="barh", color="coral", edgecolor="black")
plt.xlabel("Información Mutua")
plt.title("Ranking de características por Información Mutua\n(Target: nivel de ansiedad)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("ranking_mutual_info_anxiety.png", dpi=150)
plt.close()
print("Gráfica guardada: ranking_mutual_info_anxiety.png")


#----------FIN DEL PREPROCESADO, DISCRETIZACION Y RANKEO---------

#----------INICIO DE CONSTRUCCION Y EVALUACION DE DE MODELOS---------

# 6. PREPARACIÓN DE DATOS PARA MODELOS

print("\n" + "=" * 60)
print("  6. PREPARACIÓN DE DATOS PARA MODELOS (70/30 split)")
print("=" * 60)

# Normalizamos las características originales continuas con MinMaxScaler
# (para Gaussian NB, que asume distribución normal, funciona bien con datos escalados)
all_feat_cols = [c for c in FEATURES
                 if c not in {"anxiety_level", "self_esteem", "depression"}]

scaler = MinMaxScaler()
df_scaled = df_disc.copy()
df_scaled[all_feat_cols] = scaler.fit_transform(df_disc[all_feat_cols])

# Target: anxiety discretizado
y_anx = df_disc["anxiety_level_disc"].values

# Conjunto completo de características (sin las 3 a discretizar como raw)
X_all = df_scaled[all_feat_cols].values

# Conjunto top-k
X_top_anx = df_scaled[TOP_FEATURES_ANXIETY].values

SEED = 42
TEST_SIZE = 0.30

X_all_tr,  X_all_te,  y_anx_tr,  y_anx_te  = train_test_split(
    X_all,  y_anx, test_size=TEST_SIZE, random_state=SEED, stratify=y_anx)
X_top_tr,  X_top_te,  _, _                  = train_test_split(
    X_top_anx, y_anx, test_size=TEST_SIZE, random_state=SEED, stratify=y_anx)

print(f"\nTamaño entrenamiento: {X_all_tr.shape[0]}  |  Prueba: {X_all_te.shape[0]}")
print(f"Distribución clases ansiedad – train:\n  {np.unique(y_anx_tr, return_counts=True)}")
print(f"Distribución clases ansiedad – test :\n  {np.unique(y_anx_te, return_counts=True)}")

# 7. FUNCIÓN AUXILIAR DE EVALUACIÓN

def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_te, y_te, clases=None):
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n{'─'*50}")
    print(f"  MODELO: {nombre}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Reporte de clasificación:")
    print(classification_report(y_te, y_pred,
                                target_names=clases if clases else None,
                                zero_division=0))

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=clases if clases else "auto",
                yticklabels=clases if clases else "auto")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión\n{nombre}")
    plt.tight_layout()
    fname = nombre.lower().replace(" ", "_").replace("/", "_") + "_cm.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada: {fname}")
    return acc

CLASES_ANX = ["bajo", "medio", "alto"]

# 8. MODELOS – CLASIFICAR NIVELES DE ANSIEDAD

print("\n" + "=" * 60)
print("  8. MODELOS: CLASIFICACIÓN DE NIVELES DE ANSIEDAD")
print("=" * 60)

resultados = {}

# ── 8.1 Naive Bayes – TODAS las características ──
nb_all = GaussianNB()
acc = evaluar_modelo(
    "Naive Bayes / Todas las características (Ansiedad)",
    nb_all, X_all_tr, y_anx_tr, X_all_te, y_anx_te, CLASES_ANX)
resultados["NB_Ansiedad_Todas"] = acc

# ── 8.2 Naive Bayes – TOP-K características ──
nb_top = GaussianNB()
acc = evaluar_modelo(
    "Naive Bayes / Top características MI (Ansiedad)",
    nb_top, X_top_tr, y_anx_tr, X_top_te, y_anx_te, CLASES_ANX)
resultados["NB_Ansiedad_Top"] = acc

# ── 8.3 Árbol de Decisión – TODAS las características ──
dt_all = DecisionTreeClassifier(max_depth=6, random_state=SEED)
acc = evaluar_modelo(
    "Árbol de Decisión / Todas las características (Ansiedad)",
    dt_all, X_all_tr, y_anx_tr, X_all_te, y_anx_te, CLASES_ANX)
resultados["DT_Ansiedad_Todas"] = acc

# ── 8.4 Árbol de Decisión – TOP-K características ──
dt_top = DecisionTreeClassifier(max_depth=6, random_state=SEED)
acc = evaluar_modelo(
    "Árbol de Decisión / Top características MI (Ansiedad)",
    dt_top, X_top_tr, y_anx_tr, X_top_te, y_anx_te, CLASES_ANX)
resultados["DT_Ansiedad_Top"] = acc

# Visualizar el árbol (top-k, más legible)
plt.figure(figsize=(20, 8))
plot_tree(dt_top, feature_names=TOP_FEATURES_ANXIETY,
          class_names=CLASES_ANX, filled=True, fontsize=8, max_depth=3)
plt.title("Árbol de Decisión – Top características (Ansiedad) [profundidad máx. 3 mostrada]")
plt.tight_layout()
plt.savefig("arbol_ansiedad_top.png", dpi=120)
plt.close()
print("\nÁrbol guardado: arbol_ansiedad_top.png")

# 9. MODELOS – DETECCIÓN DE ESTRÉS (StressLevelDataset)

print("\n" + "=" * 60)
print("  9. MODELOS: DETECCIÓN DE ESTRÉS (StressLevelDataset)")
print("=" * 60)

y_stress = df_disc["stress_level"].values
CLASES_STR = ["bajo", "medio", "alto"]

X_str_tr, X_str_te, y_str_tr, y_str_te = train_test_split(
    X_all, y_stress, test_size=TEST_SIZE, random_state=SEED, stratify=y_stress)

# Ranking MI para estrés
mi_stress = mutual_info_classif(df_scaled[all_feat_cols], y_stress, random_state=42)
mi_series_str = pd.Series(mi_stress, index=all_feat_cols).sort_values(ascending=False)
TOP_FEATURES_STRESS = mi_series_str.head(TOP_K).index.tolist()
print(f"\nTop-{TOP_K} características para estrés: {TOP_FEATURES_STRESS}")

X_top_str = df_scaled[TOP_FEATURES_STRESS].values
X_top_str_tr, X_top_str_te, _, _ = train_test_split(
    X_top_str, y_stress, test_size=TEST_SIZE, random_state=SEED, stratify=y_stress)

# ── NB y DT con todas las características (estrés) ──
nb_str_all = GaussianNB()
acc = evaluar_modelo(
    "Naive Bayes / Todas las características (Estrés)",
    nb_str_all, X_str_tr, y_str_tr, X_str_te, y_str_te, CLASES_STR)
resultados["NB_Estres_Todas"] = acc

nb_str_top = GaussianNB()
acc = evaluar_modelo(
    "Naive Bayes / Top características MI (Estrés)",
    nb_str_top, X_top_str_tr, y_str_tr, X_top_str_te, y_str_te, CLASES_STR)
resultados["NB_Estres_Top"] = acc

dt_str_all = DecisionTreeClassifier(max_depth=6, random_state=SEED)
acc = evaluar_modelo(
    "Árbol de Decisión / Todas las características (Estrés)",
    dt_str_all, X_str_tr, y_str_tr, X_str_te, y_str_te, CLASES_STR)
resultados["DT_Estres_Todas"] = acc

dt_str_top = DecisionTreeClassifier(max_depth=6, random_state=SEED)
acc = evaluar_modelo(
    "Árbol de Decisión / Top características MI (Estrés)",
    dt_str_top, X_top_str_tr, y_str_tr, X_top_str_te, y_str_te, CLASES_STR)
resultados["DT_Estres_Top"] = acc

# 10. NOTA SOBRE WESAD + FACTOR DE FISHER

print("\n" + "=" * 60)
print("  10. NOTA: MODELOS WESAD (requiere dataset externo)")
print("=" * 60)
print("""
Los modelos para WESAD siguen exactamente la misma estructura
pero usando las características fisiológicas extraídas de ese dataset.
El script 'wesad_models.py' incluye:
  - Carga de features WESAD (CSV de características ya extraídas)
  - Ranking por Factor de Fisher (FLD / ratio between-within class)
  - Naive Bayes + Árbol de Decisión (todas / top-k características)
  - Split 70/30 con estratificación
  
Ejecuta: python wesad_models.py --features ruta/a/wesad_features.csv
""")

# 11. RESUMEN COMPARATIVO

print("\n" + "=" * 60)
print("  11. RESUMEN COMPARATIVO DE MODELOS")
print("=" * 60)

print(f"\n{'Modelo':<55s} {'Accuracy':>10s}")
print("-" * 67)
for nombre, acc in resultados.items():
    print(f"  {nombre:<53s} {acc:>8.4f}")

# Gráfica resumen
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(list(resultados.keys()), list(resultados.values()),
               color=["#4C72B0", "#55A868", "#C44E52", "#8172B2",
                      "#CCB974", "#64B5CD", "#4C72B0", "#55A868"],
               edgecolor="black")
ax.set_xlabel("Accuracy")
ax.set_xlim(0, 1)
ax.set_title("Comparación de modelos – Accuracy en conjunto de prueba (30%)")
ax.axvline(0.8, color="red", linestyle="--", linewidth=1, label="80% referencia")
ax.legend()
for bar, val in zip(bars, resultados.values()):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("resumen_modelos.png", dpi=150)
plt.close()
print("\nResumen guardado: resumen_modelos.png")
print("\n✓ Ejecución completada.\n")