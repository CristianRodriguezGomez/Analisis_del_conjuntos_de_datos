import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF

def extraer_datos_wesad_quest(file_path):
    datos = {}
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        lineas = f.readlines()
        
    for line in lineas:
        partes = line.replace('#', '').strip().split(';')
        if not partes: continue
        
        etiqueta = partes[0].strip()

        if etiqueta == "PANAS" and len(partes) > 10:
            datos['PANAS_Stressed'] = partes[21] if len(partes) > 21 else partes[1]
        elif etiqueta == "STAI" and len(partes) > 1:
            datos['STAI_Nervous'] = partes[2]
        elif etiqueta == "DIM" and len(partes) > 1:
            if 'SAM_Valence' not in datos:
                datos['SAM_Valence'] = partes[1]
            else:
                datos['SAM_Arousal'] = partes[1]
        elif etiqueta == "SSSQ" and len(partes) > 1:
            datos['SSSQ_Success'] = partes[2] 
            
    return datos

def generar_reporte_consolidado():
    base_path = "archive/WESAD"
    sujetos = [f'S{i}' for i in range(2, 18) if i != 12]
    
    resultados = []
    for s in sujetos:
        path = f"{base_path}/{s}/{s}_quest.csv"
        info = extraer_datos_wesad_quest(path)
        if info:
            info['Sujeto'] = s
            resultados.append(info)
    
    df_global = pd.DataFrame(resultados)
    
    cols_analisis = ['PANAS_Stressed', 'STAI_Nervous', 'SAM_Valence', 'SAM_Arousal', 'SSSQ_Success']
    for col in cols_analisis:
        df_global[col] = pd.to_numeric(df_global[col], errors='coerce')
    
    df_global.fillna(df_global.median(numeric_only=True), inplace=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols_analisis):
        df_global[col].hist(ax=axes[i], bins=5, color='orange', edgecolor='black')
        axes[i].set_title(f'Distribución {col}')
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig("survey_plots.png")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Análisis Psicométrico Global WESAD", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Courier", "", 9)
    resumen = df_global[cols_analisis].describe().round(2).to_string()
    pdf.multi_cell(0, 5, "Estadísticas todos los Sujetos):\n\n" + resumen)
    
    pdf.ln(10)
    pdf.image("survey_plots.png", x=10, w=190)
    pdf.output("Reporte_Global_Surveys.pdf")
    print("Reporte generado con éxito.")

if __name__ == "__main__":
    generar_reporte_consolidado()
