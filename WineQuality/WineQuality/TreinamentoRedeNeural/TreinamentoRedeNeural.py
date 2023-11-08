# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from google.colab import files

# Carregando o dataset
uploaded = files.upload()
data = pd.read_csv('WineQT.csv')

# Dividindo os dados em entrada (X) e saída (y)
X = data.drop('quality', axis=1) # atributos
y = data['quality'] # valor que irá prever

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definindo e compilando o modelo da rede neural
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando a rede neural
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Salvando o modelo treinado
model.save('modelo_vinho.h5')

# Salvando o objeto scaler
import pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
