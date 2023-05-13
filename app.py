import dash
from dash import html
from dash import dcc
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pandas_datareader import data as pdr

# Coleta de dados
yf.pdr_override()
symbol = 'AMZN'
start_date = '2000-01-01'
end_date = '2023-05-13'
data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)

# Pré-processamento de dados
# Use apenas o preço de fechamento
close_data = data['Close'].values.reshape(-1, 1)

# Normalize os dados
scaler = MinMaxScaler()
close_data_normalized = scaler.fit_transform(close_data)

# Crie conjuntos de treinamento e validação
train_size = int(len(close_data_normalized) * 0.8)
train_data = close_data_normalized[:train_size]
validation_data = close_data_normalized[train_size:]

# Função para criar sequências de tempo


def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)


seq_length = 30
x_train, y_train = create_sequences(train_data, seq_length)
x_validation, y_validation = create_sequences(validation_data, seq_length)

# Reformate os dados para o formato LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_validation = np.reshape(
    x_validation, (x_validation.shape[0], x_validation.shape[1], 1))

# Seleção e treinamento do modelo
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=64, epochs=10,
          validation_data=(x_validation, y_validation))

# Previsão: Preveja os próximos 30 dias
predictions = []
current_seq = x_validation[-1].reshape(1, seq_length, 1)

for _ in range(30):
    prediction = model.predict(current_seq)
    predictions.append(prediction)
    current_seq = np.roll(current_seq, -1)
    current_seq[-1][-1] = prediction

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Crie um DataFrame com as previsões
predicted_dates = pd.date_range(
    start=data.index[-1] + pd.DateOffset(1), periods=30)
predicted_prices = pd.DataFrame(
    predictions, index=predicted_dates, columns=['Predicted Close'])

# Crie as figuras com o histórico de preços e a previsão de preços
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data.index,
                          y=data['Close'],
                          mode='lines',
                          name='Histórico'))
fig1.add_trace(go.Scatter(x=predicted_prices.index,
                          y=predicted_prices['Predicted Close'],
                          mode='lines',
                          name='Previsão'))
fig1.update_layout(title='Histórico e previsão de preços da Amazon',
                   xaxis_title='Data',
                   yaxis_title='Preço (USD)',
                   xaxis_rangeslider_visible=True,
                   plot_bgcolor='#f7f7f7',
                   paper_bgcolor='#f7f7f7',
                   font=dict(color='#333'),
                   legend=dict(x=0, y=1),
                   hovermode='x unified')

# Adicione uma linha de tendência
fig1.add_trace(go.Scatter(x=data.index,
                          y=np.polyval(np.polyfit(data.index.astype(
                              'int64'), data['Close'], 1), data.index.astype('int64')),
                          mode='lines',
                          name='Tendência',
                          line=dict(color='red', width=2, dash='dash')))

# Adicione uma área sombreada
fig1.add_trace(go.Scatter(x=np.concatenate([data.index, predicted_prices.index[::-1]]),
                          y=np.concatenate(
                              [data['Close'], predicted_prices['Predicted Close'][::-1]]),
                          fill='toself',
                          fillcolor='rgba(0, 100, 80, 0.2)',
                          line=dict(color='rgba(255, 255, 255, 0)'),
                          hoverinfo='skip',
                          showlegend=False))

# Adicione um gráfico de barras
volume_data = data['Volume']
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=volume_data.index,
                      y=volume_data,
                      name='Volume',
                      marker=dict(color='rgba(0, 100, 80, 0.8)')))
fig2.update_layout(title='Volume de negociação da',
                   xaxis_title='Data',
                   yaxis_title='Volume',
                   plot_bgcolor='#f7f7f7',
                   paper_bgcolor='#f7f7f7',
                   font=dict(color='#333'))

# Crie um novo gráfico para a previsão
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=predicted_prices.index,
                          y=predicted_prices['Predicted Close'],
                          mode='lines',
                          name='Previsão'))
fig3.update_layout(title='Previsão de preços da Amazon',
                   xaxis_title='Data',
                   yaxis_title='Preço (USD)',
                   xaxis_rangeslider_visible=True,
                   plot_bgcolor='#f7f7f7',
                   paper_bgcolor='#f7f7f7',
                   font=dict(color='#333'),
                   legend=dict(x=0, y=1),
                   hovermode='x unified')

# Adicione o novo gráfico ao layout do aplicativo Dash
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Previsão de preços da Amazon',
            style={'textAlign': 'center', 'color': '#333'}),

    dcc.Graph(
        id='historico-e-previsao',
        figure=fig1
    ),

    dcc.Graph(
        id='volume-de-negociacao',
        figure=fig2
    ),

    dcc.Graph(
        id='previsao-de-precos',
        figure=fig3
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
