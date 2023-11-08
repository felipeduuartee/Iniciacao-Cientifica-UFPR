import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import StandardScaler

# Carregando o modelo treinado
model = keras.models.load_model('modelo_vinho.h5')

# Carregando o objeto scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Carregando os novos vinhos
vinhos_novos = pd.read_csv('WineQT_novos.csv')

# Normalizando os novos vinhos usando o objeto scaler
vinhos_novos = vinhos_novos.drop(columns=['quality'])
vinhos_novos_scaled = scaler.transform(vinhos_novos)

# Fazendo previsões com o modelo treinado
previsoes = model.predict(vinhos_novos_scaled)

# Convertendo previsões de probabilidades para classes
qualidades_preditas = previsoes.argmax(axis=1)

# Adicionando as qualidades preditas ao DataFrame
vinhos_novos['quality_predicted'] = qualidades_preditas

# Salvando o DataFrame atualizado
vinhos_novos.to_csv('WineQT_novos_com_predicoes.csv', index=False)
