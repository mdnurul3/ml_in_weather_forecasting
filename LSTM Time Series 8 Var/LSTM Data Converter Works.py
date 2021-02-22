import io
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weather-raw.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

plt.plot(df["Date"], df["Temp in C *10"])
plt.gcf().autofmt_xdate()
plt.show()

