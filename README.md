# Análisis de Conjuntos de Datos - WESAD y Estrés

Este proyecto implementa un flujo de trabajo para el procesamiento, limpieza y análisis exploratorio de datos (EDA) fisiológicos y psicométricos relacionados con el estrés. Se utilizan datos del dataset WESAD y conjuntos de datos complementarios.

## 📂 Estructura del Proyecto

### Scripts de Análisis
- **`leer.py`**: 
  - Procesa señales fisiológicas del pecho (ECG, EDA, EMG, Temperatura, Respiración) del dataset WESAD.
  - Realiza limpieza de datos (reemplazo de valores anormales por la mediana).
  - Genera el reporte: `Reporte_WESAD_Final.pdf` con estadísticas descriptivas e histogramas por sujeto.

- **`encuestas.py`**: 
  - Procesa los cuestionarios psicométricos (PANAS, STAI, SAM, SSSQ) de los sujetos del WESAD.
  - Genera el reporte: `Reporte_Global_Surveys.pdf` con el análisis consolidado de todos los sujetos.

- **`SSM.py`**: 
  - Analiza datasets adicionales de estrés (`StressLevelDataset.csv` y `Stress_Dataset.csv`).
  - Genera el reporte: `Analisis_Exploratorio_Ansiedad.pdf` con estadísticas y distribución de clases.

### Datos
El código espera una carpeta `archive/` con la siguiente estructura:
```text
archive/
├── StressLevelDataset.csv
├── Stress_Dataset.csv
└── WESAD/
    ├── S2/
    │   ├── S2.pkl
    │   └── S2_quest.csv
    ├── ...
```

## 🛠️ Instalación y Uso
1. **Clonar el repositorio**.
2. **Crear entorno virtual** (opcional): `python -m venv .venv`
3. **Activar entorno**: `.\.venv\Scripts\activate`
4. **Instalar librerías**: `pip install pandas matplotlib fpdf reportlab`
5. **Ejecutar scripts**:
   - `python leer.py`
   - `python encuestas.py`
   - `python SSM.py`