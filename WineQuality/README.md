# Treinamento de Rede Neural para Previsão de Qualidade de Vinhos

Este repositório contém um código Python que ilustra o treinamento de uma rede neural para prever a qualidade de vinhos. O projeto inclui a importação de bibliotecas necessárias, o carregamento de dados, a divisão dos dados em conjuntos de treinamento e teste, a normalização dos dados de entrada e a definição e treinamento do modelo da rede neural. Além disso, o modelo treinado e o objeto `scaler` utilizado para normalizar os dados são salvos para uso futuro.

## Passos do Projeto

Aqui estão os principais passos envolvidos neste projeto:

### 1. Importação de Bibliotecas

Neste primeiro passo, as bibliotecas necessárias são importadas, incluindo `pandas`, `scikit-learn`, `tensorflow` e `google.colab`. Essas bibliotecas são essenciais para carregar dados, dividir conjuntos de treinamento e teste, normalizar os dados e criar a rede neural.

### 2. Carregamento do Dataset

O conjunto de dados utilizado neste projeto é chamado de 'WineQT.csv'. Ele contém informações sobre vinhos, incluindo vários atributos e a qualidade dos vinhos. O dataset é carregado usando a função `pd.read_csv` da biblioteca pandas.

### 3. Divisão dos Dados

Os dados são divididos em duas partes: entrada (X) e saída (y). O objetivo é prever a qualidade dos vinhos com base nos atributos. Para isso, os atributos são armazenados em `X` e a qualidade em `y`. Além disso, os dados são divididos em conjuntos de treinamento e teste usando a função `train_test_split` da biblioteca scikit-learn.

### 4. Normalização dos Dados

Para garantir que a rede neural funcione eficazmente, os dados de entrada são normalizados. O objeto `StandardScaler` é usado para padronizar os valores das características.

### 5. Definição e Treinamento do Modelo

A rede neural é definida usando a biblioteca TensorFlow. Ela consiste em três camadas: duas camadas ocultas com ativação ReLU e uma camada de saída com ativação softmax para prever a qualidade dos vinhos. O modelo é compilado com a função de perda 'sparse_categorical_crossentropy' e o otimizador 'adam'. Em seguida, é treinado usando os dados de treinamento, e a validação é realizada com os dados de teste.

### 6. Salvamento do Modelo e do Objeto Scaler

Após o treinamento bem-sucedido da rede neural, o modelo treinado é salvo em um arquivo 'modelo_vinho.h5' usando a função `model.save`. Além disso, o objeto `scaler` é salvo em um arquivo 'scaler.pkl' para normalização de dados em previsões futuras.

Este README fornece uma visão geral dos passos realizados neste projeto. Certifique-se de que todas as bibliotecas estejam instaladas e de que o conjunto de dados 'WineQT.csv' esteja disponível antes de executar o código. Você pode personalizar os hiperparâmetros e a arquitetura da rede neural conforme necessário para melhorar o desempenho do modelo.
