import yfinance as yf 
import numpy as np 
import matplotlib.pyplot as plt

def total_returns(all_data, num_stocks, initial_portfolio_value):
    # Start a list that will keep track of the portfolios % change day by day
    percent_chage = []
    # for each day 
    num_days = len(all_data[0])-1
    for i in range(1, num_days):
        # declare a list that will hold all the prices for the stocks 
        new_prices = []
        # For each stock
        indx = 0
        for dataframe in all_data:
            print(f"Length of dataframe for stock {indx}: {len(dataframe)}")
            indx = indx + 1
            # grab the prices for all the stocks at the current day 
            new_prices.append(dataframe.iloc[i]["Open"])

        # calculate the new value of the portfolio
        new_value = np.dot(num_stocks, new_prices)

        # calculate the difference
        diff = new_value - initial_portfolio_value

        # calculate the percent change
        percent_chage.append(diff/initial_portfolio_value)

    return percent_chage

def daily_returns(all_data, num_stocks, initial_portfolio_value):
    # Start a list that will keep track of the portfolios % change day by day
    percent_chage = []
    # for each day 
    for i in range(1, len(all_data[0])-1):
        # declare a list that will hold all the prices for the stocks 
        new_prices = []
        # For each stock
        for dataframe in all_data:
            # grab the prices for all the stocks at the current day 
            new_prices.append(dataframe.iloc[i]["Open"])

        # calculate the new value of the portfolio
        new_value = np.dot(num_stocks, new_prices)

        # calculate the difference
        diff = new_value - initial_portfolio_value

        initial_portfolio_value = new_value
        # calculate the percent change
        percent_chage.append(diff/initial_portfolio_value)

    return percent_chage

def plot_returns(dates,drl, startindex=0):
    point = drl[startindex]
    x1 = dates[startindex]
    for i in range(startindex, len(drl)-1): 
        nextPoint  = drl[i] 
        xs = [x1, dates[i]]
        ys = [point, nextPoint]
        if nextPoint < point: 
            #negative slope 
            plt.plot(xs, ys, color='r')
        
        else: 
            #positive or horizontial slope
            plt.plot(xs,ys, color='g')
        x1 = dates[i]
        point = nextPoint

    plt.title("Daily Returns of Portfolio")
    plt.xlabel("Date")
    plt.ylabel("% Change")
    plt.show()


def return_data(drl):
    num_pos_days = 0
    num_neg_days = 0
    point = drl[0]
    for i in range(1, len(drl)-1): 
        nextPoint  = drl[i] 
        
        if nextPoint < point: 
            #negative slope 
            num_neg_days = num_neg_days + 1
        
        else: 
            #positive or horizontial slope
            num_pos_days = num_pos_days + 1

        point = nextPoint
    return num_pos_days, num_neg_days

def adjust_dataframes(all_data):
    # check to make sure the latest date matches for all stocks
    shortest_length = len(all_data[0])
    first_date = all_data[0].index[0]
    latest_date = first_date
    for j in range(1, len(all_data)):
        # if the amount of data is lower than the first (we can use < because it cannot be longer than since 
        # since we specified a start date. It can only be equal to or less than)
        current_dataframe_length = len(all_data[j])
        if current_dataframe_length < shortest_length:
            # if the lengths are different set the shortest length equal to the new length 
            shortest_length = current_dataframe_length

            # grab the starting date of the new dataframe
            latest_date = all_data[j].index[0]
            
    # make sure that all dataframes start at the correct date
    updated_data = []
    for frame in all_data:
        updated_data.append(frame.loc[latest_date:])
    
    return updated_data, latest_date

def performance(tickers, weights, init_investment, start_date, plot=False):    
    init_value = []
    num_stocks = []
    init_stock_price = []
    all_data = []

    # grab all the data we need
    for ticker in tickers: 
        all_data.append(yf.download(ticker, start=start_date,interval="1d"))
    
    
    updated_data, latest_date = adjust_dataframes(all_data)
    

    all_data = updated_data
    # grab SPY data 
    spy = yf.download("SPY", start=start_date,interval='1d')

    # start price of SPY
    spy_start_price = spy.loc[latest_date]["Open"]

    # number of shares of SPY that could be bought
    num_spy_shares = init_investment/spy_start_price

    # initial cash value
    init_spy_value = num_spy_shares * spy_start_price

    # End SPY price (most current infor)
    end_spy_price = spy.iloc[len(spy)-1]["Open"]

    # set the spy dataframe equal to the subframe that starts at the earliest date available that all stocks 
    # were available to trade. 
    spy = spy.loc[latest_date:]

    


    # for each dataframe
    indexer = 0
    for frame in all_data:
        cash_value = weights[indexer]*init_investment
        #initial stock price is of that position on the open
        init_stock_price.append(frame.iloc[0]["Open"])

        # number of stocks that could be purchased given the amount of cash for that position
        num_stocks.append(round(cash_value / frame.iloc[0]["Open"],2))

        # initial cash value (should be close to the cash value that was determined above, maybe a few cents or dollars off)
        init_value.append(num_stocks[indexer] * init_stock_price[indexer])
        # Print info
        print(f"Number of Stocks of {tickers[indexer]}: {num_stocks[indexer]} @ {round(init_stock_price[indexer],2)}/share")
        print(f"Cash Value: ${round(init_value[indexer], 2)}, Percent of Portfolio: {weights[indexer]*100}%")
        print("-------------------------------------------------")
        indexer = indexer + 1

    # Calculate change in portfolio
    # take the DOT product of the number of stocks and stock prices, this gives us the initial cash value of the portfolio
    # Again, this should be around the same amount as the initial cash invested
    initial_portfolio_value = np.dot(num_stocks,init_stock_price)

    # Start a list that will keep track of the portfolios % change day by day    
    percent_chage = total_returns(all_data, num_stocks, initial_portfolio_value)
   
    daily_returns_list = daily_returns(all_data, num_stocks, initial_portfolio_value)

  
    # calculate change in S&P 
    spy_change = []

    #for each day 
    for i in range(1, len(spy)-1):
        # get the S&P price
         next_day_price = spy.iloc[i]["Open"]

         # get the new value of the S&P position
         next_day_value = num_spy_shares * next_day_price

         # calculate the difference
         diff = next_day_value - init_spy_value

         # append
         spy_change.append(diff / init_spy_value)

    # Print S&P position information
    print(f"Number of SPY Shares: {round(num_spy_shares,2)} @ {round(spy_start_price,2)}/share")
    print(f"Cash Value: ${round(init_spy_value,2)}")
    print(f"End Cash Value: ${round(num_spy_shares * end_spy_price,2)}")
    temp1 = (((num_spy_shares * end_spy_price) - init_spy_value) / init_spy_value) * 100
    print(f"Percent Chage: {round(temp1,2)}%")
    print("-------------------------------------------------")

    # calculate change in portfolio
    print(f"Initial Portfolio Cash Value : ${round(initial_portfolio_value,2)}")

    # same as above but just for the end value
    end_prices = []
    for dataframe in all_data:
        end_prices.append(dataframe.iloc[len(dataframe)-1]["Open"])

    # calculating end value of portfolio
    endval = round(np.dot(end_prices, num_stocks),2)

    # printing information
    print(f"End Portfolio Cash Value: ${endval}")
    temp = ((endval - initial_portfolio_value) / initial_portfolio_value) * 100
    print(f"Percent Change: {round(temp,2)}%")
    print(f"Diff between S&P: {round(temp-temp1,2)}%")


    spy_indicies_to_plot = spy.index[:len(spy.index)-2]
    

    # plot results
    if plot == True:
        plt.plot(spy_indicies_to_plot, spy_change, color='b', label = "S&P 500 ETF")
        plt.plot(spy_indicies_to_plot, percent_chage, color='red', label="My Portfolio")
        plt.ylabel("% change")
        plt.xlabel("Date")
        plt.legend()
        plt.show()
        plot_returns(spy.index, daily_returns_list,500)

    pos_days, neg_days = return_data(daily_returns_list)
    print(f"Number of positive gain days: {pos_days} days")
    print(f"Number of negative gain days: {neg_days} days")

    return spy_change, percent_chage, spy.index, latest_date, daily_returns_list



stocks = ["AAPL", "SBUX", "CMG", "ADBE", "ETSY", "LULU", "META", "NFLX", "PINS", "PTON", "SHOP", "AMZN"]
weights = [0.20,  0.15,    0.05, 0.01,   0.01,   0.10,  0.05,      0.1,    0.05,    0.05,  0.05,  0.18  ]

cash = 10000 

date = '2019-01-01'


spyc, pc, dates, startdate , drl = performance(stocks, weights, cash, date, plot=True)
