import yahooquery
import pandas as pd
from yahooquery import Ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.fft import fft
import math
matplotlib.use('Qt5Agg')


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

stock = Ticker('djia')

# Default period = ytd, interval = 1d
df = stock.history(period="max", interval='1d')
df.head()

prices = list(df["adjclose"])

data = pd.DataFrame(np.diff(df["adjclose"].values)) #these are the delta values


#finding the amplitude
amplitude = 0
data_list = data[0].values.tolist()
for point in data_list:
    if point > amplitude:
        amplitude = point


plt.plot([i for i in range(len(data))], data, markersize=2)
plt.title("Stock Delta Value")
plt.xlabel("Day")
plt.ylabel("Change in Price")
plt.show()

X = fft(data)
N = len(data)
n = np.arange(N)
sample_rate = 1/1 #1 sample / 1 second (treating a day as a second)
T = N/(sample_rate)  # number of samples over the rate of sampling (1 sample/ day)
freq = n/T

### seems to not work as well when have big jump, need to find a reasonable amplitude
### taking the graph that shows the frequencies and their amplitudes and trying to determine which one is dominant
dom_freq = 0
largest_xval = 0
for val in range(len(X)):
    if abs(X[val]) > largest_xval:
        largest_xval = abs(X[val])
        dom_freq = freq[val]

#if one were interested in finding the average frequecny:
# avg_freq = 0
# for sig in freq:
#     avg_freq += sig
# avg_freq /= len(freq)
# print(avg_freq)


plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.title("Frequency Dominance")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 1)
plt.show()

x = np.linspace(0, [i for i in range(len(data))],251)
plt.plot(x, amplitude/2*np.sin(dom_freq*x), 'b-', label='y=sin(x)') #divide amplitude by 2 for more "regular" looking wave
plt.plot([i for i in range(len(data))], data, markersize=2)
plt.title("Stock Delta Value")
plt.xlabel("Day")
plt.ylabel("Change in Price")
plt.show()