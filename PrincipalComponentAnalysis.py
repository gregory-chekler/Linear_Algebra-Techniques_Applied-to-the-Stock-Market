import pandas as pd
from yahooquery import Ticker
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from operator import add
matplotlib.use('Qt5Agg')


def change_in_price(ticker):
    stock = Ticker(str(ticker))
    df = stock.history(period="1y", interval='1d')
    df.head()
    prices = list(df['adjclose'])
    change = []
    for i in range(1, len(prices)):
        change.append(prices[i] - prices[i-1])
    return change

stocks = pd.DataFrame()


### stocks that make up the Dow Jones Industrial average (except DOW b/c doesn't work)
stock_tickers = ['aapl', 'msft', 'unh', 'v', 'jnj', 'wmt', 'jpm',
                 'pg', 'hd', 'cvx', 'ko', 'dis', 'csco', 'vz',
                 'nke', 'mrk', 'intc', 'crm', 'mcd', 'axp',
                 'amgn', 'hon', 'cat', 'ibm', 'gs', 'ba', 'mmm',
                 'trv', 'wba']


## by getting the change in stock, we are nomralizing the data and putting it closer to the x axis
## maybe get a graph between the stocks vs change in price to see difference
for stock in stock_tickers:
    stocks[stock] = change_in_price(stock)



covariance = stocks.cov() # the covariance matrix, finds correlation btw stocks

eigenvals, eigenvectors = np.linalg.eigh(covariance)

eigenvals = list(reversed(list(eigenvals)))

plt.plot([i for i in range(len(eigenvals))], eigenvals, markersize=2)
plt.title("Principle components")
plt.xlabel("Eigenvals")
plt.ylabel("Value")
plt.show()




## want to find when there is a drop off to the principal components
## also notice the orthogonality of all eigenvectors
## can do this by looking when the change in value btwn eigenvals is less than 10%

def diff(curr, prev):
    return math.fabs((curr - prev)/prev)*100

principle_eigenvals = []
for i in range(1, len(eigenvals)):
    if diff(eigenvals[i], eigenvals[i-1]) > 10:
        principle_eigenvals.append(eigenvals[i-1])
    else:
        break

principle_components = eigenvectors[len(eigenvectors)-len(principle_eigenvals): len(eigenvectors)] ##eignevectors are in reverse of importance to need to go to last one
np.flip(principle_components, axis=1)


plt.bar(stock_tickers, principle_components[0])
plt.title("1st principle component")
plt.xlabel("Stocks")
plt.ylabel("Values")
plt.show()

plt.bar(stock_tickers, principle_components[1])
plt.title("2nd principle component")
plt.xlabel("Stocks")
plt.ylabel("Values")
plt.show()


### We have seen how the first and second principle components vary in their weighting,
### it should be interesting to see the weightings play out
### to this end, let us create portfolios consisting of positive weightings of the principle components


portfolio_one = []
for i in range(len(principle_components[0])):
    if principle_components[0][i] > 0:
        portfolio_one.append(stock_tickers[i])

portfolio_two = []
for i in range(len(principle_components[1])):
    if principle_components[1][i] > 0:
        portfolio_two.append(stock_tickers[i])

neg_portfolio_1 = []
for i in range(len(principle_components[0])):
    if principle_components[0][i] < 0:
        neg_portfolio_1.append(stock_tickers[i])

neg_portfolio_2 = []
for i in range(len(principle_components[1])):
    if principle_components[1][i] < 0:
        neg_portfolio_2.append(stock_tickers[i])

print(portfolio_one,
      portfolio_one,
      neg_portfolio_1,
      neg_portfolio_2)

### we can make a psuedo-index fund that is sort of the like the DJI ()

def fund(portfolio):
    tot_price = 252*[0.0] ## need to remeber weekends don't count ##22 for month/252 for year
    for ticker in portfolio:
        stock = Ticker(str(ticker))
        df = stock.history(period="1y", interval='1d')
        df.head()
        prices = list(df['adjclose'])

        tot_price = list(map(add, prices, tot_price))


    avg_tot_price= []
    portfolio_size = len(portfolio)
    for i in range(len(tot_price)):
        avg_tot_price.append(tot_price[i]/portfolio_size)
    return avg_tot_price

print(np.matmul(principle_components, covariance))

actual_DJI = fund(stock_tickers)
fund1 = fund(portfolio_one)
fund2 = fund(portfolio_two)
neg_fund1 = fund(neg_portfolio_1)
neg_fund2 = fund(neg_portfolio_2)


plt.plot([i for i in range(len(actual_DJI))], actual_DJI, markersize=2, label="Actual DJI")
plt.plot([i for i in range(len(fund1))], fund1, markersize=2, label='fund1')
plt.plot([i for i in range(len(fund2))], fund2, markersize=2, label="fund2")
plt.plot([i for i in range(len(neg_fund1))], neg_fund1, markersize=2, label='neg_fund1')
plt.plot([i for i in range(len(neg_fund2))], neg_fund2, markersize=2, label='neg_fund2')
plt.title("Averaged Funds price over a year")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()