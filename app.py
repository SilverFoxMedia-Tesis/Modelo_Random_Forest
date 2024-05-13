from flask import Flask, request, jsonify
import joblib
import numpy as np


def add_noise(data_array, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data_array.shape)
    return data_array * (1 + noise)

# Cargar los modelos de comidas
models = {}
output_columns = [
    "desayuno_1",
    "desayuno_2",
    "desayuno_3",
    "almuerzo_1",
    "almuerzo_2",
    "almuerzo_3",
    "cena_1",
    "cena_2"
]
for col in output_columns:
    models[col] = joblib.load(f'{col}_model.pkl')

# Cargar el mapeo de etiquetas para comidas
output_mappings = joblib.load('output_mappings.pkl')

# Cargar los modelos de ejercicios
exercise_models = {}
exercise_output_columns = ["ejercicio_1", "ejercicio_2", "ejercicio_3", "ejercicio_4", "ejercicio_5"]
for col in exercise_output_columns:
    exercise_models[col] = joblib.load(f'{col}_model.pkl')

# Cargar el mapeo de etiquetas para ejercicios
exercise_mappings = joblib.load('exercise_mappings.pkl')

# Inicializa la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Extraer los datos del request en formato JSON
    data = request.json

    # Verificar que se tengan todas las columnas necesarias
    food_input_columns = [
        "calorieDeficit",
        "desayunoCarbs",
        "desayunoProten",
        "desayunoGrasas",
        "almuerzoCarbs",
        "almuerzoProten",
        "almuerzoGrasas",
        "cenaCarbs",
        "cenaProten",
        "cenaGrasas"
    ]
    exercise_input_columns = ["abdominalCircumference", "physicalActivity"]

    # Convertir los datos de comidas en un array para predicciones
    try:
        food_input_data = np.array([data[col] for col in food_input_columns]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Falta la columna {str(e)} en la solicitud de comidas'}), 400

    # Convertir los datos de ejercicios en un array para predicciones
    try:
        exercise_input_data = np.array([data[col] for col in exercise_input_columns]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Falta la columna {str(e)} en la solicitud de ejercicios'}), 400

    # Aplicar ruido para añadir variabilidad
    food_input_data = add_noise(food_input_data)
    exercise_input_data = add_noise(exercise_input_data)

    # Realizar las predicciones para cada columna objetivo de comidas
    predictions = {}
    for col in output_columns:
        prediction_idx = models[col].predict(food_input_data)[0]
        predictions[col] = list(output_mappings[col])[prediction_idx]

    # Realizar las predicciones para cada columna objetivo de ejercicios
    for col in exercise_output_columns:
        prediction_idx = exercise_models[col].predict(exercise_input_data)[0]
        predictions[col] = list(exercise_mappings[col])[prediction_idx]

    # print(predictions)
  
    return jsonify({"predictions": predictions})

# Ejecutar el servidor en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, port=5000)
