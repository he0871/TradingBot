latest_price = window_data.iloc[-1]["TSLA"]
predicted_price = predict_next_price(model, window_data)

if predicted_price > latest_price * 1.001:
    signal = "buy"
elif predicted_price < latest_price * 0.999:
    signal = "sell"
else:
    signal = "hold"
