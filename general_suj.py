import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from fpdf import FPDF


# ================================
# REPORTE PDF
# ================================

class ReporteWESAD(FPDF):

    def header(self):
        self.set_font('Arial','B',16)
        self.cell(0,10,'Reporte Analisis Dataset WESAD',0,1,'C')
        self.ln(5)


# ================================
# LIMPIEZA OUTLIERS
# ================================

def limpiar_outliers(df):

    filtros = {
        'Temp': (20,45),
        'Resp': (-35,35),
        'ECG': (-1.5,1.5),
        'EDA': (0,60)
    }

    for col, rango in filtros.items():

        if col in df.columns:

            lim_inf, lim_sup = rango

            mask = (df[col] < lim_inf) | (df[col] > lim_sup)

            mediana = df.loc[~mask, col].median()

            df.loc[mask, col] = mediana

    return df


# ================================
# FISHER SCORE (STRESS VS NO STRESS)
# ================================

def fisher_score(X, y):

    scores = {}

    for col in X.columns:

        stress = X.loc[y == 1, col]
        no_stress = X.loc[y == 0, col]

        mean1 = stress.mean()
        mean2 = no_stress.mean()

        var1 = stress.var()
        var2 = no_stress.var()

        fisher = ((mean1 - mean2)**2) / (var1 + var2) if (var1 + var2) != 0 else 0

        scores[col] = fisher

    fisher_df = pd.DataFrame.from_dict(
        scores,
        orient='index',
        columns=['Fisher']
    )

    fisher_df = fisher_df.sort_values(by='Fisher', ascending=False)

    return fisher_df


# ================================
# FORWARD SELECTION
# ================================

def forward_selection(X, fisher_df, k=5, alpha1=0.7, alpha2=0.3):

    selected = []

    fisher_dict = fisher_df['Fisher'].to_dict()

    first = fisher_df.index[0]

    selected.append(first)

    while len(selected) < k:

        best_score = -np.inf
        best_feature = None

        for feature in X.columns:

            if feature in selected:
                continue

            fisher = fisher_dict.get(feature, 0)

            corr = 0

            for s in selected:
                corr += abs(X[s].corr(X[feature]))

            corr = corr / len(selected)

            score = alpha1 * fisher - alpha2 * corr

            if score > best_score:
                best_score = score
                best_feature = feature

        selected.append(best_feature)

    return selected


# ================================
# CARGAR DATASET
# ================================

def cargar_dataset():

    base_path = "archive/WESAD"

    dataset_total = []

    sujetos = [s for s in os.listdir(base_path) if s.startswith('S')]
    sujetos.sort(key=lambda x: int(x[1:]))

    for sujeto in sujetos:

        path = f"{base_path}/{sujeto}/{sujeto}.pkl"

        if not os.path.exists(path):
            continue

        print("Cargando", sujeto)

        with open(path,'rb') as f:
            data = pickle.load(f, encoding='latin1')

        signals = data['signal']['chest']

        df_list = []

        for name, values in signals.items():

            if len(values.shape) > 1 and values.shape[1] > 1:

                cols = [f"{name}_{i}" for i in range(values.shape[1])]
                df_temp = pd.DataFrame(values, columns=cols)

            else:

                df_temp = pd.DataFrame(values, columns=[name])

            df_list.append(df_temp)

        df = pd.concat(df_list, axis=1)

        labels = pd.Series(data['label'], name="label")

        df['label'] = labels[:len(df)]

        df['Subject'] = sujeto

        df = df[df['label'] != 0]

        df = limpiar_outliers(df)

        dataset_total.append(df)

    dataset_total = pd.concat(dataset_total)

    return dataset_total


# ================================
# GRAFICAS EDA
# ================================

def graficas_eda(df):

    features = df.drop(columns=['label','Subject'])

    features.hist(figsize=(15,10), bins=50)
    plt.tight_layout()
    plt.savefig("histogramas.png")
    plt.close()

    plt.figure(figsize=(15,8))
    features.boxplot(rot=90)
    plt.tight_layout()
    plt.savefig("boxplots.png")
    plt.close()

    corr = features.corr()

    plt.figure(figsize=(10,8))
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Matriz de Correlacion")

    plt.tight_layout()
    plt.savefig("correlacion.png")
    plt.close()

    return corr


# ================================
# REPORTE PDF
# ================================

def generar_reporte(df, corr,
                    fisher_global, forward_global,
                    tabla_fisher_subject, tabla_forward_subject):

    pdf = ReporteWESAD()

    pdf.set_auto_page_break(auto=True, margin=15)

    # Estadísticas
    pdf.add_page()
    pdf.set_font('Arial','B',14)
    pdf.cell(0,10,"Estadisticas Descriptivas",0,1)

    desc = df.describe().round(4).T

    pdf.set_font('Courier','',9)
    pdf.multi_cell(0,5, desc.to_string())

    # Histogramas
    pdf.add_page()
    pdf.cell(0,10,"Histogramas",0,1)
    pdf.image("histogramas.png", x=10, w=190)

    # Boxplots
    pdf.add_page()
    pdf.cell(0,10,"Boxplots",0,1)
    pdf.image("boxplots.png", x=10, w=190)

    # Correlación
    pdf.add_page()
    pdf.cell(0,10,"Matriz de Correlacion",0,1)
    pdf.image("correlacion.png", x=10, w=180)

    # =========================
    # ANALISIS GLOBAL
    # =========================

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

    # =========================
    # ANALISIS POR SUJETO
    # =========================

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

    pdf.output("Reporte_Final_WESAD.pdf")


# ================================
# MAIN
# ================================

def main():

    print("Cargando dataset...")
    df = cargar_dataset()

    print("Dataset size:", df.shape)

    print("Generando EDA...")
    corr = graficas_eda(df)

    # =========================
    # ANALISIS GLOBAL
    # =========================

    X = df.drop(columns=['label','Subject'])

    y = df['label'].apply(lambda x: 1 if x == 2 else 0)

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    fisher_global = fisher_score(X_scaled, y)

    forward_global = forward_selection(X_scaled, fisher_global)

    print("\nRanking Fisher GLOBAL")
    print(fisher_global.head())

    print("\nTop 5 GLOBAL")
    print(forward_global)

    # =========================
    # ANALISIS POR SUJETO
    # =========================

    subjects = df['Subject'].unique()

    tabla_fisher = []
    tabla_forward = []

    for s in subjects:

        print("\nSUJETO", s)

        df_s = df[df['Subject'] == s]

        X = df_s.drop(columns=['label','Subject'])

        y = df_s['label'].apply(lambda x: 1 if x == 2 else 0)

        scaler = StandardScaler()

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        fisher = fisher_score(X_scaled, y)

        for f,score in fisher['Fisher'].items():
            tabla_fisher.append([s,f,score])

        selected = forward_selection(X_scaled, fisher)

        for f in selected:
            tabla_forward.append([s,f])

    tabla_fisher = pd.DataFrame(
        tabla_fisher,
        columns=["Subject","Feature","Fisher Score"]
    )

    tabla_forward = pd.DataFrame(
        tabla_forward,
        columns=["Subject","Selected Feature"]
    )

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


if __name__ == "__main__":
    main()