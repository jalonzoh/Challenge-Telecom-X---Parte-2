# Challenge-Telecom-X---Parte-2
Challenge Telecomx 2 - Creación de un modelo de predicción (Evasoión de Clientes - churn)

## Índice 📋

1. Descripción del proyecto.
2. Acceso al proyecto
3. Etapas del proyecto.
4. Descripción de los datos
5. Resultados y conclusiones
6. Tecnologías utilizadas.
7. Agradecimientos.
8. Desarrollador del proyecto.

## 1. Descripción del proyecto 📚

* Se desarrollaron varios modelos supervisados de Machine Learning para identificar clientes propensos a cancelar el servicio en una empresa de telecomunicaciones, el modelo con mejor desempeño a través de métricas de evaluación específicas sera le ganador.
* Los datos se ajustaron a los requerimientos de cada algoritmo, teniendo en cuanta la sensibilidad  y la multicolinealidad entre variables.
* Optimización de hiperparámetros (dentro cada conjunto de datos), seleccionando el mejor modelo por cada conjunto de datos para luego compararlos entre sí. En la selección final priorizó la métrica **Recall**, a fin de minimizar los falsos negativos (clientes que abandonan y no son detectados), sin comprometer la **Precisión**.
* Finalmente, en un entorno de simulación productiva con datos sintéticos se valido la operación del modelo.

## Acceso al proyecto 📂

Para obtener acceso el proyecto hacer lo siguiente:

1. Puedes descargarlo directamente desde el repositorio en GitHub en el siguiente enlace:
   <p><a href="https://github.com/jalonzoh/Challenge-Telecom-X---Parte-2">https://github.com/jalonzoh/Challenge-Telecom-X---Parte-2</p>

Descargar un archivo comprimido `.zip`.


## 3. Etapas del proyecto 📝

1. Descripción del proyecto
2. Importación de librerías y configuraciones
   - Importación de librerias
   - Paths
   - Configuraciones
   - Funciones
3. Preprocesamiento de datos
   - Encoding de variables categóricas
   - Normalizacion de datos
   - Correlacion entre variables
   - Análisis de multicolinealidad
   - Análisis dirigido
4. Modelado de datos
   - Train Test split
   - Escalado de variables numéricas
   - Balance del dataset
   - Baseline Model - Decision Tree Classifier
   - Random Forest Classifier
   - Logistic Regression
   - K-Nearest Neighbors
   - XGBoost Classifier
   - Support Vector Machine
6. Evaluación Best Models
   - Métricas Generales
   - Subajuste (Underfitting) y Sobreajuste (Overfitting)
   - Matrices de confusión
   - Importancias y Coeficientes
7. Champion Model
8. Pipeline de prueba en entorno productivo
   - Generación de datos artificiales
   - Pipeline de prueba

## 4. Descripción de los datos 📊

En la etapa anterior, se realizó exploración y limpieza de los datos, obteniendo dos conjuntos de datos:
* <a href="ttps://raw.githubusercontent.com/jalonzoh/Challenge-Telecom-X---Parte-2/refs/heads/main/TelecomX_dataLimpio.json">TelecomX_dataLimpio.json</a>
* <a href="https://raw.githubusercontent.com/jalonzoh/Challenge-Telecom-X---Parte-2/refs/heads/main/no_clientes_importantes.json">no_clientes_importantes.json</a>

Ambos archivos se unieron en uno solo dando **7152 registros**. El archivo **no_clientes_importantes.json** contiene clientes que abandonaron la empresa en la etapa anterior, se incluyen en este proyecto para tener un escenario real del comportamiento de los clientes.

### Variables

| Variable           | Tipo       | Descripción breve                         | Valores originales                             | Preprocesado          |
| ------------------ | ---------- | ----------------------------------------- | ---------------------------------------------- | --------------------- |
| `customerID`       | Categórica | Identificador único del cliente           | String                                         | -                     |
| `Gender`           | Categórica | Género del cliente                        | `'Male'`, `'Female'`                           | One-hot-encoding      |
| `SeniorCitizen`    | Categórica | Indica si el cliente es mayor de 65 años  | `0`, `1`                                       | One-hot-encoding      |
| `Partner`          | Categórica | Si el cliente tiene pareja                | `'Yes'`, `'No'`                                | One-hot-encoding      |
| `Dependents`       | Categórica | Si el cliente tiene personas a cargo      | `'Yes'`, `'No'`                                | One-hot-encoding      |
| `PhoneService`     | Categórica | Si tiene servicio telefónico              | `'Yes'`, `'No'`                                | One-hot-encoding      |
| `MultipleLines`    | Categórica | Si tiene múltiples líneas telefónicas     | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `InternetService`  | Categórica | Tipo de conexión a internet               | `'DSL'`, `'Fiber optic'`, `'No'`               | One-hot-encoding      |
| `OnlineSecurity`   | Categórica | Seguridad en línea                        | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `OnlineBackup`     | Categórica | Respaldo en línea                         | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `DeviceProtection` | Categórica | Protección de dispositivo                 | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `TechSupport`      | Categórica | Soporte técnico                           | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `StreamingTV`      | Categórica | TV en streaming                           | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `StreamingMovies`  | Categórica | Películas en streaming                    | `'Yes'`, `'No'`,                               | One-hot-encoding      |
| `Contract`         | Categórica | Tipo de contrato                          | `'Month-to-month'`, `'One year'`, `'Two year'` | One-hot-encoding      |
| `PaperlessBilling` | Categórica | Si el cliente usa facturación electrónica | `'Yes'`, `'No'`                                | One-hot-encoding      |
| `PaymentMethod`    | Categórica | Método de pago                            | 4 categorías                                   | One-hot-encoding      |
| `Tenure`           | Numérica   | Antigüedad en meses                       | int, `0` a `72`                                | Igual/Escalado        |
| `ChargesMonthly`   | Numérica   | Costo mensual del servicio                | float                                          | Igual/Escalado        |
| `ChargesTotal`     | Numérica   | Costo total acumulado del cliente         | float                                          | Igual/Escalado        |
| `ChargesDaily`     | Numérica   | Estimación diaria del costo del cliente   | float (`Charges.Monthly/30`)                   | Descartada            |
| `Churn`            | Categórica | Si el cliente abandonó la empresa         | `'Yes'`, `'No'`                                | Label Encoding        |

### Balance de clases

Al ver un desbalance en la variable de respuesta (`Churn`), se hizo una reducción del Dataset utilizando `NearMiss Version 3`.
Se redujo la clase mayoritaria para que los modelos contubieran datos reales y no generar sesgos. 

Esta reducción dio como resultado un conjunto de datos con **3362 registros** para el entrenamiento, y conservando la distribución original de los datos para la evaluación de modelos, con un total de **1073 registros**, de lo cual **72.3%** son etiquetados como `Churn = 0` (clase mayoritaria) y **27.7%** etiquetados como `Churn = 1` (clase minoritaria).

Para la simulación del pipeline en productivo, se generaro datos artificiales utilizando la técnica `SMOTENC`.

### Codificación y reescalado de datos

* Modelos basados en árboles: `Random Forest Classifier` y `XGBoost Classifier`
  - Codificación `One-hot`, descartando una variable cuando esta era de naturaleza binaria (dos categorías) a través del parámetro `drop='if_binary'`.
  - Variables numéricas: `Tenure`, `ChargesMonthly` y `ChargesTotal` no fueron escaladas ya que estos modelos no se ven afectados por la escala de los datos debido a su naturaleza basada en particiones y árboles.

* Modelos lineales: `Logistic Regression` y `Support Vector Machine (kernel = 'linear')`
  - Codificación `One-hot` para variables categóricas con el parámetro `drop='first'` descartando la primera categoría de cada variable para evitar introducir multicolinealidad al modelo.
  - Variables numéricas: escaladas utilizando `Robust Scaler` debido a la presencia de valores atípicos, el cual utiliza la mediana y el rango intercuartílico (IQR) para escalar los datos (lo cual lo hace resistente a outliers), debido a la sensibilidad de dichos modelos a la escala de los datos.
  - Dataset: `X_linear`

* Modelos basados en distancia: `K-Nearest Neighbors Classifier`
  - Codificación `One-hot` con el parámetro `drop='if_binary'`, ya que este tipo de modelos se beneficia de mayor cantidad de variables al calcular la distancia entre observaciones.
  - Variables numéricas escaladas con `Robust Scaler`, ya que el modelo es sensible a la escala de los datos.
  - Dataset: `X_scaled`
 
* La variable respuesta (Churn) fue codificada utilizando `Label Encoder`, transformando:
  - `'Yes'` -> `1`
  - `'No'` -> `0`

<br><br>
## 5. Resultados y conclusiones ✍️

### Modelos

Al evaluar los modelos de cada familia para seleccionar un modelo campeón (**Champion Model**), se obtubo los siguientes resultados de los modelos:

| Modelo                       | Dataset                      | Accuracy | Precision | Recall | F1-score | AUC    | Umbral |
|------------------------------|------------------------------|----------|-----------|--------|----------|--------|--------|
| Best Random Forest           | X                            | 0.7698   | 0.5654    | 0.7273 | 0.6362   | 0.8326 | 0.5    |
| Best Logistic Regression     | X_linear[selected_features]  | 0.7633   | 0.5657    | 0.6229 | 0.5929   | 0.8137 | 0.5    |
| Best K-Nearest Neighbors     | X_linear                     | 0.7251   | 0.5027    | 0.6364 | 0.5617   | 0.7513 | 0.5    |
| Best XGBoost Classifier      | X                            | 0.7568   | 0.5464    | 0.7138 | 0.6190   | 0.8188 | 0.5    |
| Best Support Vector Machine  | X_linear                     | 0.7540   | 0.5563    | 0.5488 | 0.5525   | 0.8073 | 0.5    |

Se buscó llevar la métrica **Recall** a un valor mínimo de 0.85 al modificar el umbral de decisión del modelo:

| Modelo                       | Dataset                      | Accuracy | Precision | Recall | F1-score | AUC    | Umbral |
|------------------------------|------------------------------|----------|-----------|--------|----------|--------|--------|
| Best Random Forest           | X                            | 0.7148   | 0.4914    | 0.8620 | 0.6259   | 0.8326 | 0.39   |
| Best Logistic Regression     | X_linear[selected_features]  | 0.6747   | 0.4539    | 0.8620 | 0.5947   | 0.8137 | 0.39   |
| Best XGBoost Classifier      | X                            | 0.6925   | 0.4695    | 0.8552 | 0.6062   | 0.8188 | 0.38   |
| Best Support Vector Machine  | X_linear                     | 0.6785   | 0.4571    | 0.8620 | 0.5974   | 0.8073 | 0.38   |

Para los resultados de las métricas obtenidas, se seleccionó el modelo `Best Random Forest` como **Champion Model** para implementación en entorno productivo, esto debido a su capacidad de generalización.

A pesar de mostrar cierto sobreajuste a los datos de entrenamiento, como puede verse en las tablas a continuación, `Best Random Forest` se mantiene como el mejor generalizador.


| Model	                       | Recall Train	 | Recall Test	     | Variación   |
|-------------------------------|----------------|------------------|-------------|
| Best Random Forest	           | 0.8769	       | 0.7273	        | -14.96%     |
| Best Logistic Regression	     | 0.6383	       | 0.6229	        | -1.54%      | 
| Best XGBoost Classifier	     | 0.7555	       | 0.7138	        | -4.17%      | 
| Best Support Vector Machine   | 0.5996	       | 0.5488	        | -5.08%      | 

| Model	                       | F1-score Train	 | F1-score Test	     | Variación   |
|-------------------------------|-------------------|---------------------|-------------|
| Best Random Forest	           | 0.8673	          | 0.6362              | -23.11%     |
| Best Logistic Regression	     | 0.6301	          | 0.5929	           | -3.72%      |
| Best XGBoost Classifier	     | 0.7591	          | 0.6190	           | -14.01%     |
| Best Support Vector Machine	  | 0.6126	          | 0.5525	           | -6.01%      |

### Pipeline de prueba

Se desarrolló la simulación de un pipeline para la implementación del modelo en entorno productivo, utilizando datos sintéticos generados con la técnica `SMOTENC`.
El mismo, recibe un archivo JSON  con datos sin ninguna transformación para producir predicciones.
Cuenta con dos modos de utilización:

* `mode='production'`: que devuelve un archivo JSON con `CustomerID`, `Probabilidad Churn` y `Churn` *(Etiqueta: si Probabilidad Churn >= 0.39, Churn = 1, si Probabilidad Churn < 0.39, Churn = 0)*
* `mode='monitor'`: devuelve un archivo JSON con un campo con fecha y hora de ejecución del monitoreo (`Model`), sus métricas `Accuracy`, `Precision`, `Recall` y `F1-score`, para umbral de decisión por defecto y umbral de decisión modificado, y tiempo de predicción.

Dicho pipeline realiza las transformaciones necesarias sobre los datos crudos utilizando los artefactos creados a lo largo del proyecto.

Todo esto permitió construir un modelo predictivo sólido y su aplicabilidad real en entornos simulados.

## 6. Tecnologías utilizadas 🛠️

* `Python-3.11.7`
* `Numpy-1.26.4`
* `pandas-2.2.2`
* `matplotlib-3.10.0`
* `seaborn-0.13.2`
* `scikit--learn-1.5.2`
* `imbalanced--learn-1.5.2`
* `xgboost-2.1.3`
* `Google Colab`
* `Git and GitHub`

## 7. Agradecimientos 🤝

A Oracle y Alura LATAM 

## 8. Desarrollador 👷

**| Jhon Alonzo | Data Scientist Junior |**
