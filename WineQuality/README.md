# Repositório de Previsão de Qualidade de Vinhos com Rede Neural

Bem-vindo ao Repositório de Previsão de Qualidade de Vinhos com Rede Neural. Este repositório contém dois códigos Python interligados que permitem treinar uma rede neural para prever a qualidade de vinhos e, em seguida, usar o modelo treinado para prever a qualidade de novos vinhos. 

## Treinamento da Rede Neural

O primeiro código, localizado no diretório `TreinamentoRedeNeural`, é responsável pelo treinamento da rede neural. Aqui estão os principais passos realizados neste código:

1. Importação de Bibliotecas: As bibliotecas necessárias, como `pandas`, `scikit-learn`, e `tensorflow`, são importadas para carregar dados, dividir conjuntos de treinamento e teste, normalizar os dados e criar a rede neural.

2. Carregamento do Dataset: Um conjunto de dados chamado 'WineQT.csv' é carregado, contendo informações sobre vinhos, incluindo atributos e qualidades.

3. Divisão dos Dados: Os dados são divididos em atributos (X) e qualidades (y) para treinar o modelo. Os dados também são divididos em conjuntos de treinamento e teste.

4. Normalização dos Dados: Os dados de entrada são normalizados para garantir que a rede neural funcione eficazmente.

5. Definição e Treinamento do Modelo: O modelo da rede neural é definido e compilado, seguido pelo treinamento com os dados de treinamento.

6. Salvamento do Modelo e do Objeto Scaler: O modelo treinado é salvo em 'modelo_vinho.h5', e o objeto scaler é salvo em 'scaler.pkl' para uso futuro.

## Uso do Modelo Treinado para Prever Qualidade de Novos Vinhos

O segundo código, localizado no diretório `AplicacaoRedeNeural`, permite usar o modelo de rede neural treinado para prever a qualidade de novos vinhos. Aqui estão os passos envolvidos:

1. Carregamento do Modelo Treinado e do Objeto Scaler: O modelo previamente treinado é carregado a partir do arquivo 'modelo_vinho.h5', e o objeto `scaler` usado para normalizar os dados de entrada é carregado a partir de 'scaler.pkl'.

2. Carregamento dos Novos Vinhos: Os dados dos novos vinhos são carregados a partir de 'WineQT_novos.csv' para realizar previsões.

3. Normalização dos Novos Vinhos: Os novos dados de vinhos são normalizados com o objeto `scaler`.

4. Previsões com o Modelo Treinado: O modelo é utilizado para prever a qualidade dos novos vinhos.

5. Adição das Qualidades Preditas ao DataFrame: As qualidades previstas são adicionadas ao DataFrame original como uma nova coluna.

6. Salvamento do DataFrame Atualizado: O DataFrame atualizado com as previsões é salvo em 'WineQT_novos_com_predicoes.csv'.

## Como Usar Este Repositório

1. Clone este repositório para o seu ambiente local.

2. Certifique-se de que as bibliotecas necessárias estão instaladas e de que os conjuntos de dados estão disponíveis no mesmo diretório.

3. Execute o código `treinamento_rede_neural.py` para treinar a rede neural e salvar o modelo.

4. Execute o código `uso_modelo_treinado.py` para usar o modelo treinado e prever a qualidade de novos vinhos.

Certifique-se de personalizar os nomes dos arquivos de entrada e saída, se necessário. Este repositório é uma demonstração de como treinar uma rede neural para previsão de qualidade de vinhos e usá-la para fazer previsões em novos dados. Você pode adaptar e expandir o projeto de acordo com suas necessidades.
