import matplotlib.pyplot as plt 
import numpy as np 
import yfinance as yf

def performance(tickers, weights, init_investment, start_date):    
    init_value = []
    num_stocks = []
    init_stock_price = []
    all_data = []

    # grab all the data we need
    for ticker in tickers: 
        all_data.append(yf.download(ticker, start=start_date,interval="1d"))
    
    
        
    # check to make sure the latest date matches for all stocks
    first_stock_length = len(all_data[0])
    first_date = all_data[0].index[0]
    latest_date = first_date
    shortest_inx = 0
    for j in range(1, len(all_data)):
        if len(all_data[j]) != first_stock_length:
            # there is a difference in stock lengths
            latest_date = max(all_data[j].index[0], first_date)
            shortest_inx = j

    # grab SPY data 
    spy = yf.download("SPY", start=start_date,interval='1d')
    spy_start_price = spy.loc[latest_date]["Open"]
    num_spy_shares = init_investment/spy_start_price
    init_spy_value = num_spy_shares * spy_start_price
    end_spy_price = spy.iloc[len(spy)-1]["Open"]
    spy = spy.loc[latest_date:]
    # now we get the cash value 
    for i in range(len(all_data)):
        cash_value = weights[i]*init_investment
        init_stock_price.append(all_data[i].loc[latest_date]["Open"])
        num_stocks.append(round(cash_value/all_data[i].loc[latest_date]["Open"],2))
        init_value.append(num_stocks[i] * init_stock_price[i])
        print(f"Number of Stocks of {tickers[i]}: {num_stocks[i]} @ {round(init_stock_price[i],2)}/share")
        print(f"Cash Value: {round(init_value[i], 2)}, Percent of Portfolio: {weights[i]*100}%")
        print("-------------------------------------------------")

    # Calculate change in portfolio
    initial_portfolio_value = np.dot(num_stocks,init_stock_price)
    percent_chage = []
    for i in range(len(all_data[shortest_inx])-2):
        # we need to get the value of the portfolio compared to the inital portfolio value
        new_prices = []
        for dataframe in all_data:
            new_prices.append(dataframe.iloc[i+1]["Open"])
        new_value = np.dot(num_stocks, new_prices)
        diff = new_value - initial_portfolio_value
        percent_chage.append(diff/initial_portfolio_value)

    # Print S&P position information
    print(f"Number of SPY Shares: {round(num_spy_shares,2)} @ {round(spy_start_price,2)}/share")
    print(f"Cash Value: {round(init_spy_value,2)}")
    print(f"End Cash Value: {round(num_spy_shares * end_spy_price,2)}")
    temp1 = (((num_spy_shares * end_spy_price) - init_spy_value) / init_spy_value) * 100
    print(f"Percent Chage: {round(temp1,2)}")


    # calculate change in S&P 
    spy_change = []
    for i in range(len(spy)-2):
         next_day_price = spy.iloc[i+1]["Open"]
         next_day_value = num_spy_shares * next_day_price
         diff = next_day_value - init_spy_value
         spy_change.append(diff / init_spy_value)
    print("-------------------------------------------------")

    # calculate change in portfolio
    print(f"Initial Portfolio Cash Value : {round(initial_portfolio_value,2)}")
    end_prices = []
    for dataframe in all_data:
        end_prices.append(dataframe.iloc[len(dataframe)-1]["Open"])
    endval = round(np.dot(end_prices, num_stocks),2)
    print(f"End Portfolio Cash Value: {endval}")
    temp = ((endval - initial_portfolio_value) / initial_portfolio_value) * 100
    print(f"Percent Change: {round(temp,2)}")
    print(f"Diff between S&P: {round(temp-temp1,2)}")


    # plot results
    plt.plot(spy.index[:len(spy.index)-2],percent_chage, color='red', label="My Portfolio")
    plt.plot(spy.index[:len(spy.index)-2], spy_change, color='b', label = "S&P 500 ETF")
    plt.ylabel("% change")
    plt.xlabel("Date")
    plt.legend()
    plt.show()
    return spy_change, percent_chage, spy.index, latest_date
    