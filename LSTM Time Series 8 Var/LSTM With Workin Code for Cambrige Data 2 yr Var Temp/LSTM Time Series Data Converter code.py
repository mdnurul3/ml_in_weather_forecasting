import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

df = pd.read_excel('weather-raw.xls')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M%S %p")
print(type(df))
plt.plot(df["Date"], df["Pressure (mBar)"])
plt.gcf().autofmt_xdate()
plt.show()
# save to file
df.to_csv('pollution_weather_raw.csv')
print(df.head(2))