import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Predicción de Producción Futura en Panadería")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.write(data.head())
    
    # Seleccionar columnas predictoras y columna objetivo
    st.write("Selecciona las columnas para entrenar el modelo")
    
    # Mostrar las columnas disponibles
    columnas = list(data.columns)
    columnas_predictoras = st.multiselect("Selecciona las columnas predictoras", columnas)
    col_y = st.selectbox("Selecciona la columna objetivo (a predecir)", columnas)
    
    if st.button("Entrenar Modelo"):
        if len(columnas_predictoras) == 0 or col_y is None:
            st.error("Por favor, selecciona al menos dos columnas predictoras y una columna objetivo.")
        else:
            # Separar las variables predictoras y la variable objetivo
            X = data[columnas_predictoras]
            y = data[col_y]
            
            # Convertir variables categóricas en variables numéricas (si es necesario)
            X = pd.get_dummies(X, drop_first=True)
            
            # Dividir los datos en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar un modelo de regresión lineal
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Hacer predicciones y mostrar métricas de evaluación
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)

            st.write(f"Error Cuadrático Medio (MSE): {mse}")
            st.write(f"Error Absoluto Medio (MAE): {mae}")
            
            # Guardar el modelo entrenado en la app
            st.session_state['model'] = model
            st.session_state['columns'] = X.columns  # Guardar las columnas de X
            st.success("Modelo entrenado correctamente")
            
            # Gráfico de comparación entre valores reales y predichos
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, color='blue', label='Predicciones')
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Valores Reales')
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Valores Reales vs Predicciones')
            plt.legend()
            st.pyplot(plt)

# Predicción con el modelo entrenado
if 'model' in st.session_state:
    st.write("Introduce los valores para realizar una predicción:")
    
    # Crear campos dinámicos para los inputs de predicción
    inputs = {}
    for col in st.session_state['columns']:
        val = st.number_input(f"Introduce el valor para {col}")
        inputs[col] = val
    
    if st.button("Predecir"):
        # Convertir los inputs en formato adecuado para el modelo
        input_data = pd.DataFrame([inputs])
        input_data = pd.get_dummies(input_data, drop_first=True)
        input_data = input_data.reindex(columns=st.session_state['columns'], fill_value=0)
        
        # Hacer la predicción
        prediccion = st.session_state['model'].predict(input_data)
        st.success(f"La predicción es: {prediccion[0]}")
        
        # Explicación de la predicción
        st.write("### Explicación de los resultados")
        st.write(f"El valor predicho representa la producción diaria en función de los factores seleccionados. "
                 f"Por ejemplo, si estás prediciendo `produccion_diaria`, este valor te da una estimación de la cantidad de panes que se espera producir "
                 f"basado en las variables que ingresaste (como temperatura del horno, cantidad de harina, etc.).")
