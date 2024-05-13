import requests

# Datos de entrada para comidas y ejercicios
data = {
    "calorieDeficit": 1522.4,
    "desayunoCarbs": 47.575,
    "desayunoProten": 19.03,
    "desayunoGrasas": 12.686667,
    "almuerzoCarbs": 66.605,
    "almuerzoProten": 26.642,
    "almuerzoGrasas": 17.761333,
    "cenaCarbs": 76.12,
    "cenaProten": 30.448002,
    "cenaGrasas": 20.298668,
    "abdominalCircumference": 71,  # Circunferencia abdominal
    "physicalActivity": 1          # Nivel de actividad física (1-5, por ejemplo)
}

# Enviar solicitud POST
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Mostrar la respuesta
print(response.json())
