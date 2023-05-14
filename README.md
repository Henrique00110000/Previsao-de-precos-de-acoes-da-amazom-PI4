# Previsao-de-precos-de-acoes-da-amazom-PI4

Previsão de Preços de Ações da Amazon
Este projeto utiliza um modelo LSTM para prever preços futuros das ações da Amazon. Os dados históricos de preços são coletados da Yahoo Finance usando o pandas_datareader, pré-processados com o MinMaxScaler do Scikit-learn e treinados em um modelo LSTM usando o Sequential do Keras. As previsões são visualizadas usando o plotly.graph_objects e exibidas em um aplicativo da web criado com o framework Dash.

Como executar
Para executar o aplicativo da web, siga os seguintes passos:

Clone ou faça o download deste repositório para sua máquina.
Instale as bibliotecas necessárias executando o seguinte comando em um terminal ou prompt de comando:
pip install dash numpy pandas pandas-datareader plotly scikit-learn tensorflow yfinance

Navegue até o diretório em que você baixou/clonou o repositório e execute o seguinte comando:
python app.py

Isso iniciará um servidor local na porta 8050. Abra um navegador da web e navegue até http://localhost:8050 para visualizar o aplicativo.
Bibliotecas utilizadas
dash
numpy
pandas
pandas-datareader
plotly
scikit-learn
tensorflow
yfinance
Créditos
Este projeto foi criado por Henrique. Ele foi inspirado no projeto Stock Price Prediction with LSTM no Kaggle.

Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.
