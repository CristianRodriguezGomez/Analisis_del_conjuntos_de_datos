import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import re
# Funciones auxiliares
def limpiar_nombre(nombre):
    # Reemplaza todo lo que no sea letra o número por _
    return re.sub(r'[^a-zA-Z0-9]', '_', nombre)

def guardar_histogramas(df, columnas, prefijo):
    rutas = []
    for col in columnas:
        plt.figure()
        plt.hist(df[col], bins=5)
        plt.title(f'Histograma de {col}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')

        nombre_limpio = limpiar_nombre(col)
        ruta = f"{prefijo}_{nombre_limpio}.png"

        plt.savefig(ruta, bbox_inches="tight")
        plt.close()
        rutas.append(ruta)

    return rutas
def guardar_histograma_clase(serie, titulo, nombre):
    plt.figure()
    serie.value_counts().sort_index().plot(kind="bar")
    plt.title(titulo)
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.savefig(nombre, bbox_inches="tight")
    plt.close()
    return nombre


def tabla_descriptiva_a_parrafos(desc, styles):
    elementos = []
    for col in desc.columns:
        texto = f"<b>{col}</b><br/>"
        for idx in desc.index:
            texto += f"{idx}: {round(desc[col][idx], 3)}<br/>"
        elementos.append(Paragraph(texto, styles["Normal"]))
        elementos.append(Spacer(1, 10))
    return elementos

# Cargar datasets
df1 = pd.read_csv("archive/StressLevelDataset.csv")
df2 = pd.read_csv("archive/Stress_Dataset.csv")
# Crear PDF
pdf = SimpleDocTemplate("Analisis_Exploratorio_Ansiedad.pdf", pagesize=A4)
styles = getSampleStyleSheet()
contenido = []

# DATASET 1
contenido.append(Paragraph("Dataset 1: StressLevelDataset", styles["Title"]))
contenido.append(Spacer(1, 12))

filas, columnas = df1.shape
contenido.append(Paragraph(f"Número de registros: {filas}", styles["Normal"]))
contenido.append(Paragraph(f"Número de columnas: {columnas}", styles["Normal"]))
contenido.append(Spacer(1, 12))

contenido.append(Paragraph("Tipos de datos (todas las variables son numéricas)", styles["Heading2"]))
contenido.append(Paragraph(str(df1.dtypes), styles["Normal"]))
contenido.append(Spacer(1, 12))

contenido.append(Paragraph("Estadísticas descriptivas", styles["Heading2"]))
desc1 = df1.describe()
contenido.extend(tabla_descriptiva_a_parrafos(desc1, styles))

# Histogramas Dataset 1
cols1 = df1.columns.drop("stress_level")
imgs1 = guardar_histogramas(df1, cols1, "ds1")
img_clase1 = guardar_histograma_clase(
    df1["stress_level"],
    "Histograma de clase: Stress Level",
    "ds1_stress.png"
)

contenido.append(PageBreak())
contenido.append(Paragraph("Histogramas - Dataset 1", styles["Heading2"]))
contenido.append(Spacer(1, 12))

for img in imgs1 + [img_clase1]:
    contenido.append(Image(img, width=400, height=300))
    contenido.append(Spacer(1, 12))

contenido.append(PageBreak())

# DATASET 2
contenido.append(Paragraph("Dataset 2: Stress_Dataset", styles["Title"]))
contenido.append(Spacer(1, 12))

filas, columnas = df2.shape
contenido.append(Paragraph(f"Número de registros: {filas}", styles["Normal"]))
contenido.append(Paragraph(f"Número de columnas: {columnas}", styles["Normal"]))
contenido.append(Spacer(1, 12))

contenido.append(Paragraph("Tipos de datos (numéricos y categóricos)", styles["Heading2"]))
contenido.append(Paragraph(str(df2.dtypes), styles["Normal"]))
contenido.append(Spacer(1, 12))

contenido.append(Paragraph("Estadísticas descriptivas (variables numéricas)", styles["Heading2"]))
desc2 = df2.describe()
contenido.extend(tabla_descriptiva_a_parrafos(desc2, styles))

# Variable categórica Dataset 2

contenido.append(Spacer(1, 12))
contenido.append(Paragraph(
    "Descripción de la variable categórica: Which type of stress do you primarily experience?",
    styles["Heading2"]
))

frecuencia = df2["Which type of stress do you primarily experience?"].value_counts()
porcentaje = df2["Which type of stress do you primarily experience?"].value_counts(normalize=True) * 100
moda = df2["Which type of stress do you primarily experience?"].mode()[0]

texto_cat = "<b>Frecuencia:</b><br/>"
for idx, val in frecuencia.items():
    texto_cat += f"{idx}: {val}<br/>"

texto_cat += "<br/><b>Porcentaje:</b><br/>"
for idx, val in porcentaje.items():
    texto_cat += f"{idx}: {round(val,2)}%<br/>"

texto_cat += f"<br/><b>Moda:</b> {moda}"

contenido.append(Paragraph(texto_cat, styles["Normal"]))
contenido.append(Spacer(1, 12))

# Histogramas Dataset 2
cols2 = df2.columns.drop("Which type of stress do you primarily experience?")
imgs2 = guardar_histogramas(df2, cols2, "ds2")

labels_limpias = df2["Which type of stress do you primarily experience?"].apply(lambda x: x.split()[0])
img_clase2 = guardar_histograma_clase(
    labels_limpias,
    "Histograma de clase: Tipo de Estrés",
    "ds2_stress.png"
)

contenido.append(PageBreak())
contenido.append(Paragraph("Histogramas - Dataset 2", styles["Heading2"]))
contenido.append(Spacer(1, 12))

for img in imgs2 + [img_clase2]:
    contenido.append(Image(img, width=400, height=300))
    contenido.append(Spacer(1, 12))

# Generar PDF

pdf.build(contenido)

print("PDF generado correctamente: Analisis_Exploratorio_Ansiedad.pdf")
