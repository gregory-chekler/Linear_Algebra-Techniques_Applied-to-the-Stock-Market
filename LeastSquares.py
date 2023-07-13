import numpy
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import yahooquery

stock_tickers = ['aapl', 'msft', 'unh', 'v', 'jnj', 'wmt', 'jpm',
                 'pg', 'hd', 'cvx', 'ko', 'dis', 'csco', 'vz',
                 'nke', 'mrk', 'intc', 'crm', 'mcd', 'axp',
                 'amgn', 'hon', 'cat', 'ibm', 'gs', 'mmm',
                 'trv'] # left out 'ba' and 'wba'

Financials_Matrix = []
prices_vector = []

for stock in stock_tickers:
    ticker = yahooquery.Ticker(stock)
    stock_info = ticker.quotes

    uppercase_stock = str(stock).upper()

    ## we are going to want:
    financial_data = []
    financial_data.append(stock_info[uppercase_stock]['regularMarketPreviousClose'])
    financial_data.append(stock_info[uppercase_stock]['averageDailyVolume10Day'])
    financial_data.append(stock_info[uppercase_stock]['marketCap'])
    financial_data.append(stock_info[uppercase_stock]['trailingPE'])
    financial_data.append(stock_info[uppercase_stock]['sharesOutstanding'])



    df = ticker.history(period="1d", interval='1d')
    prices_vector.append(df['adjclose']) # need current prices matrix to be the b matrix

    Financials_Matrix.append(financial_data)

# setting up A^TA =A^Tb equation
prices_vector = np.matrix(prices_vector)
financials_matrix = np.matrix(Financials_Matrix)
financials_matrix_transpose = np.transpose(financials_matrix)

#solving equation
left_side = np.matmul(financials_matrix_transpose, financials_matrix)
right_side = np.matmul(financials_matrix_transpose, prices_vector)

coefficients = numpy.linalg.solve(left_side, right_side)

# getting coefficients
prev_close = coefficients[0]
avg_vol = coefficients[1]
market_cap = coefficients[2]
trailing_price_to_earnings = coefficients[3]
shares_outstanding = coefficients[4]

print("The coefficient associated with previous close price was:", prev_close,
      ";\naverage volume traded in the past 10 days was:", avg_vol, ";\nmarket cap was: ", market_cap,
      ";\ntrailing price to earnings was:", trailing_price_to_earnings, ":\nshares outstanding was:", shares_outstanding)


#### lets try out least squares on some stocks to see what it would predict vs reality
test_stocks = ['tsla', 'crox', 'foxa', 'lmt']

print("Let us try our determined coefficients on some other stocks to see how effective it is.")

for stock in test_stocks:
    ticker = yahooquery.Ticker(stock)
    stock_info = ticker.quotes

    uppercase_stock = str(stock).upper()

    ## we are going to want:

    financial_data = []
    financial_data.append(stock_info[uppercase_stock]['regularMarketPreviousClose'])
    financial_data.append(stock_info[uppercase_stock]['averageDailyVolume10Day'])
    financial_data.append(stock_info[uppercase_stock]['marketCap'])
    financial_data.append(stock_info[uppercase_stock]['trailingPE'])
    financial_data.append(stock_info[uppercase_stock]['sharesOutstanding'])



    df = ticker.history(period="5d", interval='1d')
    today_price = list(df['adjclose'])[0]
    financial_data = np.matrix(financial_data)
    predicted = np.matmul(financial_data, coefficients)

    print("For", stock, "the predicted price was:", predicted, "and the actual price is:", today_price)


