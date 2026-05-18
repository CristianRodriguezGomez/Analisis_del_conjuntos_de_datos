# Clasificación de Ansiedad y Detección de Estrés mediante Aprendizaje Supervisado

[cite_start]Este repositorio contiene la implementación integral del proyecto de Reconocimiento de Patrones[cite: 808, 809, 944]. [cite_start]Se enfoca en el procesamiento, limpieza, análisis exploratorio (EDA) y entrenamiento de modelos predictivos utilizando datos psicométricos, contextuales y señales biomédicas dinámicas[cite: 818, 955]. [cite_start]El objetivo central es evaluar el rendimiento de los algoritmos **Bayes Ingenuo** y **Árboles de Decisión** en tareas de clasificación multiclase y binaria[cite: 944, 951].

---

##  Integrantes
* [cite_start]Arce Jiménez Anelí [cite: 813, 947]
* [cite_start]Acevedo Herrera Ossiel Alejandro [cite: 814, 948]
* [cite_start]Aragón Toledo José Ramón [cite: 814, 948]
* [cite_start]Rodríguez Gómez Cristian [cite: 814, 948]

---
## Adquisicion y Configuracion del Dataset WESAD

> **Nota Critica sobre el Volumen de Datos:** Debido a que el dataset WESAD (Wearable Stress and Affect Detection) contiene senales fisiologicas continuas de alta resolucion, su tamano aproximado es de 17.64 GB. Por motivos de almacenamiento y cuotas del repositorio, estos archivos no estan incluidos en el codigo fuente y deben descargarse manualmente.

### Pasos para la Configuracion Local:

1. **Descarga del Dataset:** Obten el archivo comprimido oficial desde la plataforma Kaggle o el servidor de la Universidad de Stuttgart bajo el nombre de Wearable Stress and Affect Detection (WESAD).
2. **Creacion del Directorio de Almacenamiento:** En la raiz de este proyecto, crea una carpeta llamada obligatoriamente `archive/`.
3. **Extraccion de Estructuras Binarias (.pkl):**
   * Descomprime el archivo ZIP descargado dentro de la carpeta `archive/`.
   * Asegurate de que se conserve la carpeta interna `WESAD/` conteniendo los directorios de cada individuo (S2, S3, ..., S17) con sus respectivos archivos binarios `.pkl` (sincronizados a 700 Hz) y sus cuestionarios `_quest.csv`.

### Estructura de Caminos (Paths) Esperada por los Scripts:
Los archivos de codigo (`final_wesad.py`, `leer.py`, `encuestas.py`) resolveran las rutas relativas de manera automatica siempre y cuando el arbol de directorios este organizado de la siguiente forma:

```text
PATRONES-PARCIAL2/
├── archive/
│   └── WESAD/
│       ├── S2/
│       │   ├── S2.pkl
│       │   └── S2_quest.csv
│       ├── S3/
│       └── ...
├── final_wesad.py
├── SSM.py
├── encuestas.py
├── StressLevelDataset.csv
└── leer.py

```

##  Estructura del Proyecto y Scripts Centrales


### 1. Análisis Tabular y Psicométrico (`StressLevelDataset.csv`)
* **`SSM.py`**: 
  * [cite_start]Ejecuta el pipeline completo de procesamiento de datos para la clasificación de niveles de ansiedad (Variable Objetivo / Target: `anxiety_level`)[cite: 982, 1034].
  * [cite_start]Divide el dataset de manera estratificada en **70% Entrenamiento y 30% Prueba** para aislar estrictamente la evaluación[cite: 1252, 1257, 1260].
  * [cite_start]Implementa una rutina de **Discretización ($T=3$)** basada en rangos e intervalos aprendidos únicamente en el conjunto de entrenamiento para prevenir de manera estricta la **Fuga de Datos (Data Leakage)**[cite: 829, 1146].
  * [cite_start]Realiza la **Selección de Características** mediante el cálculo exacto de la **Información Mutua ($I(X; Y)$)** fundamentada en la Entropía de Shannon[cite: 818, 831, 1164, 1166].
  * [cite_start]Entrena y evalúa los modelos de Bayes Ingenuo (`CategoricalNB`) y Árboles de Decisión (`DecisionTreeClassifier`) comparando el rendimiento de todas las variables contra el Top 5 seleccionado[cite: 944, 1266, 1267, 1273, 1274].

### 2. Procesamiento de Señales Biomédicas (`WESAD Dataset`)
* **`final_wesad.py`**:
  * Funciona como el script unificado de extracción, limpieza y modelado para el conjunto de datos de sensores portátiles.
  * [cite_start]Procesa y limpia señales en bruto extraídas del pecho (dispositivo *RespiBAN* a 700 Hz) enfocándose en **ECG (Electrocardiograma)** y **EDA (Actividad Electrodérmica)**[cite: 1366, 1431, 1432].
  * [cite_start]Ejecuta un re-etiquetado binario de las actividades para clasificar los estados en **No-Estrés (0)** y **Estrés (1)** e implementa una ventana temporal de 120 segundos con 50% de solapamiento[cite: 843, 845, 856, 868, 1428, 1437, 1438].
  * Transforma los datos crudos a características estadísticas del dominio del tiempo y frecuencia (conteo de picos fásicos `SCR_count`, promedio de intervalos `MeanNN`, variabilidad global `SDNN`, entre otros)[cite: 818, 837, 1443, 1447, 1449].
  * [cite_start]Aplica estandarización por **Z-score** para resolver la disparidad de escalas y utiliza el **Factor de Fisher** para el ranking y selección de los mejores atributos discriminantes[cite: 818, 1484, 1485, 1526, 1528].

### 3. Scripts Auxiliares y Documentación
* **`encuestas.py`**: Procesa los cuestionarios psicométricos contextuales y consolida el análisis exploratorio preliminar.
* [cite_start]**`leer.py`**: Script de inspección inicial y lectura rápida de estructuras `.pkl` nativas del dataset WESAD[cite: 1356, 1390].
* **`atributos_wesad.xlsx`**: Matriz de datos complementaria utilizada para mapear las propiedades estadísticas de las señales biomédicas procesadas.

---

## Instalación y Uso

### 1. Clonar el repositorio y preparar el entorno
Asegúrate de ejecutar los comandos desde la raíz del proyecto para que las rutas relativas se resuelvan correctamente.

```bash
# Crear el entorno virtual controlado
python -m venv venv

# Activar el entorno virtual (Windows)
.\venv\Scripts\activate

# Activar el entorno virtual (Linux/macOS)
source venv/bin/activate