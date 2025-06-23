# Proyecto de Predicción de Señales de Trading con Regresión Logística

Este proyecto explora cómo se puede utilizar Machine Learning para construir un modelo de predicción de señales de trading basadas en datos del ETF SPY. A través del uso de regresión logística y técnicas de ingeniería de características, buscamos predecir si el rendimiento de un activo será positivo o negativo en el corto plazo.

## Parte 1: Pipeline inicial de ML

- Se cargó un dataset histórico de SPY con precios, volumen y una señal de trading como target.
- Se generaron y agregaron al menos 6 indicadores técnicos:
  - Media Móvil Simple (SMA)
  - Media Móvil Exponencial (EWMA)
  - Volatilidad Móvil
  - RSI (Índice de Fuerza Relativa)
  - Cambio porcentual en volumen
  - Bandas de Bollinger
  - Momentum

- Se definieron los conjuntos de features (X) y el target (y).
- Se dividió el dataset en 70% entrenamiento y 30% prueba.
- Se aplicó preprocesamiento con:
  - `OneHotEncoder` para variables categóricas.
  - `StandardScaler` para variables numéricas.

- Se construyó un pipeline de `scikit-learn` con:
  - Preprocesamiento
  - Balanceo con SMOTE
  - Modelo de Regresión Logística

- Se entrenó y evaluó el modelo con métricas como:
  - Accuracy, Precision, Recall, F1-Score
  - Matriz de confusión

## Parte 2: Pipeline con PCA, Cross-Validation y GridSearch

- Se utilizó PCA para reducir la dimensionalidad explicando al menos el 80% de la varianza.
- Se aplicó `GridSearchCV` para buscar los mejores hiperparámetros.
- Se integró `StratifiedKFold` como técnica de validación cruzada.
- Se repitió el entrenamiento y evaluación, y se compararon resultados con los obtenidos en la Parte 1.

## Parte 3: Reflexión Estratégica

- Se analizó el impacto de Falsos Positivos y Falsos Negativos en un contexto real de trading.
- Se argumentó cuál métrica (precisión, recall, F1, etc.) sería la más relevante para una firma financiera.
- Se concluyó sobre las ventajas de utilizar modelos de aprendizaje automático en comparación con estrategias técnicas puras.

---

## ¿Cómo correr el proyecto?

1. Tener un entorno de Python 3 instalado (ya sea `venv` o `conda`).
2. Instalar las dependencias necesarias con:

   ```bash
   pip install -r requirements.txt

3. Abrir el archivo Jupyter Notebook
    ```bash
    jupyter notebook Entrega_3.ipynb

4. Ejecutar cada celda en orden para visualizar resultados, métricas, gráficas y conclusiones.