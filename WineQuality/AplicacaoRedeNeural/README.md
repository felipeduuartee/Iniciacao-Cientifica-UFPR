# Uso do Modelo Treinado para Prever a Qualidade de Novos Vinhos

Este repositório contém um código Python para usar um modelo de rede neural previamente treinado para prever a qualidade de novos vinhos. O projeto inclui a importação de bibliotecas necessárias, o carregamento do modelo treinado, a normalização dos novos dados de vinhos, a previsão das qualidades e a adição das previsões ao DataFrame original. Além disso, o DataFrame atualizado é salvo em um arquivo CSV.

## Passos do Projeto

Aqui estão os principais passos envolvidos neste projeto:

### 1. Importação de Bibliotecas

Neste primeiro passo, as bibliotecas necessárias são importadas, incluindo `pandas`, `tensorflow`, `pickle` e `scikit-learn`. Essas bibliotecas são essenciais para carregar o modelo treinado, normalizar os novos dados e realizar previsões.

### 2. Carregamento do Modelo Treinado e do Objeto Scaler

O modelo de rede neural previamente treinado é carregado a partir do arquivo 'modelo_vinho.h5' usando a função `keras.models.load_model`. Além disso, o objeto `scaler` utilizado para normalizar os dados de entrada é carregado a partir do arquivo 'scaler.pkl' usando a biblioteca `pickle`.

### 3. Carregamento dos Novos Vinhos

Os dados dos novos vinhos são carregados a partir do arquivo 'WineQT_novos.csv' usando a função `pd.read_csv`. Esses dados são necessários para realizar previsões de qualidade.

### 4. Normalização dos Novos Vinhos

Os novos dados de vinhos são normalizados utilizando o objeto `scaler` previamente carregado. As colunas de qualidade são removidas dos novos dados, uma vez que serão previstas pelo modelo. Os dados normalizados são armazenados em `vinhos_novos_scaled`.

### 5. Previsões com o Modelo Treinado

O modelo treinado é usado para fazer previsões de qualidade dos novos vinhos. As previsões são obtidas com a função `model.predict` aplicada aos dados normalizados.

### 6. Conversão de Previsões em Classes

As previsões, que são originalmente probabilidades, são convertidas em classes de qualidade. Isso é feito selecionando a classe com maior probabilidade para cada vinho.

### 7. Adição das Qualidades Preditas ao DataFrame

As qualidades preditas são adicionadas ao DataFrame original 'vinhos_novos' como uma nova coluna chamada 'quality_predicted'. Isso permite comparar as qualidades reais com as previstas.

### 8. Salvamento do DataFrame Atualizado

O DataFrame atualizado com as previsões é salvo em um arquivo CSV chamado 'WineQT_novos_com_predicoes.csv' usando a função `to_csv`. Isso possibilita a análise e revisão das previsões.

Este README fornece uma visão geral dos passos realizados neste projeto. Certifique-se de que o modelo treinado ('modelo_vinho.h5') e o objeto scaler ('scaler.pkl') estejam disponíveis antes de executar o código. Além disso, o conjunto de dados 'WineQT_novos.csv' deve estar presente para fazer previsões. Certifique-se de ajustar o nome do arquivo de entrada e saída, se necessário.
