import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from fpdf import FPDF

#Configuración del reporte
class ReporteWESAD(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Análisis Estadístico Completo - WESAD', 0, 1, 'C')
        self.ln(5)

def generar_reporte():
    pdf = ReporteWESAD()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Lista de sujetos
    base_path = "archive/WESAD"
    sujetos = [s for s in os.listdir(base_path) if s.startswith('S') and os.path.isdir(os.path.join(base_path, s))]
    sujetos.sort(key=lambda x: int(x[1:])) 

    for sujeto in sujetos:
        file_path = f"{base_path}/{sujeto}/{sujeto}.pkl"
        if not os.path.exists(file_path): continue

        print(f"Procesando {sujeto}...")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # 1. Construcción de la Tabla
        signals = data['signal']['chest']
        df_list = []
        for s_name, s_data in signals.items():
            # Manejo de nombres de columnas para ACC (X, Y, Z) y otros
            if s_data.ndim > 1 and s_data.shape[1] > 1:
                cols = [f"{s_name}_{axis}" for axis in ['X', 'Y', 'Z']]
            else:
                cols = [s_name]
            df_list.append(pd.DataFrame(s_data, columns=cols))
        
        df = pd.concat(df_list, axis=1)

        # --- BLOQUE DE LIMPIEZA PARA ELIMINAR DATOS ANORMALES ---
        # --- EN CASO DE ENCONTRAR DATO ANORMAL SE REEMPLAZA POR LA MEDIANA ---
        filtros = {
            'Temp': (20, 45),    # Temperatura piel humana
            'Resp': (-35, 35),   # Rango normal de expansión pecho
            'ECG': (-1.5, 1.5),  # Voltaje cardíaco estándar
            'EDA': (0, 60)       # Conductancia (nunca negativa)
        }

        for col, rango in filtros.items():
            if col in df.columns:
                lim_inf, lim_sup = rango
                mask_error = (df[col] < lim_inf) | (df[col] > lim_sup)
                
                # Calculamos mediana solo de los valores que están dentro del rango
                mediana_sana = df.loc[~mask_error, col].median()
                
                # Aplicamos el reemplazo
                df.loc[mask_error, col] = mediana_sana
        # --- FIN DE LIMPIEZA ---
        
        # 2. Descripción con Redondeo
        # .round(4) quita el exceso de dígitos
        desc_df = df.describe().round(4).T 
        desc_text = desc_df.to_string()

        # 3. Agregar al PDF
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Sujeto: {sujeto} - Estadísticas Descriptivas", 0, 1)
        pdf.ln(2)
        
        pdf.set_font('Courier', '', 9)
        pdf.multi_cell(0, 5, desc_text)
        pdf.ln(10)

        # 4. Generar Histogramas por cada columna
        # Creamos una cuadrícula de subplots para que todos quepan en una imagen
        num_cols = len(df.columns)
        fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(12, 15))
        axes = axes.flatten()

        for i, col_name in enumerate(df.columns):
            df[col_name].hist(ax=axes[i], bins=50, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribución {col_name}')
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Frecuencia')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        hist_path = f"hists_{sujeto}.png"
        plt.savefig(hist_path)
        plt.close()

        # 5. Insertar los histogramas en el PDF
        pdf.image(hist_path, x=15, w=180)
        
        # Limpieza para la siguiente iteración
        os.remove(hist_path)
        del data, df, df_list

    pdf.output("Reporte_WESAD_Final.pdf")

if __name__ == "__main__":
    generar_reporte()
