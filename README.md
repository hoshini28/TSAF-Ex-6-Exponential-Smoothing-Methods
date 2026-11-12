# TSAF-Ex-6-Exponential-Smoothing-Methods

## AIM:
To implement and compare three exponential smoothing techniques for time-series
forecasting:

1. Simple Exponential Smoothing

2. Holt's Linear Trend Method

3. Holt-Winters Seasonal Method


## PROGRAM:

```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# 1. Simple Exponential Smoothing (Stable inventory)
units_stable = [500, 520, 510, 515, 495, 525, 508, 518, 502, 512]
data1 = pd.Series(units_stable)
model1 = SimpleExpSmoothing(data1)
fit1 = model1.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)
print("1. Inventory Demand Forecast (SES):")
print(forecast1)

# 2. Holt's Linear Method (Rising product demand)
units_trend = [500, 540, 585, 630, 675, 720, 770, 815, 860, 910]
data2 = pd.Series(units_trend)
model2 = Holt(data2)
fit2 = model2.fit()
forecast2 = fit2.forecast(3)
print("\n2. Rising Product Demand Forecast (Holt):")
print(forecast2)

# 3. Holt-Winters Method (Quarterly seasonality = 4 periods)
units_seasonal = [
 400, 650, 900, 550, # Year 1
 430, 690, 950, 580, # Year 2
 460, 730, 990, 620, # Year 3
 490, 780, 1040, 650 # Year 4
]
data3 = pd.Series(units_seasonal)
model3 = ExponentialSmoothing(data3, trend="add", seasonal="add", seasonal_periods=4)
fit3 = model3.fit()
forecast3 = fit3.forecast(4)
print("\n3. Quarterly Inventory Forecast (Holt-Winters):")
print(forecast3)

```

## OUTPUT:

<img width="492" height="327" alt="image" src="https://github.com/user-attachments/assets/16816b67-868c-44ee-a16b-718b0cf649d3" />
