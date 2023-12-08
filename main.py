import streamlit as st
from datetime import date


import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "MSFT", "WMT")
selected_stock = st.selectbox("Select stock for prediction.", stocks)

n_years = st.slider("Year:", 1, 4)
period = n_years * 365


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name= 'stock_open' ))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name= 'stock_close' ))     
    fig.layout.update(title_text = "Time series data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Prediciton

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
prediction = m.predict(future)

st.subheader("Prediction Data")
st.write(prediction.tail())

st.write("Predicition Graph")
fig1 = plot_plotly(m, prediction)
st.plotly_chart(fig1)

st.write("Prediciton components")
fig2 = m.plot_components(prediction)
st.write(fig2)


print(st.__version__)
print(yf.__version__)
print(Prophet.__version__)

