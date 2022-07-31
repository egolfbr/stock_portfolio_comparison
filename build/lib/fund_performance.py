import matplotlib.pyplot as plt 
import numpy as np 
import yfinance as yf
import pandas as pd

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
    cash_values = []
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
        cash_values.append(new_value)

        # calculate the difference
        diff = new_value - initial_portfolio_value

        initial_portfolio_value = new_value
        # calculate the percent change
        percent_chage.append(diff/initial_portfolio_value)

    return percent_chage, cash_values

def plot_cash_value(cash_values, spvalue, dates): 
    plt.plot(dates, cash_values, label='Portfolio Cash Value')
    plt.plot(dates, spvalue, label='S&P 500 Cash Value')
    plt.xlabel("Dates")
    plt.ylabel("Cash Value of Portfolio vs S&P 500")
    plt.legend()
    plt.title("Cash Value of Portfolio")
    plt.show()

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

def performance(tickers, weights, init_investment, start_date, plot=False, daily_return_plot_start_idx = 0):    
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
   
    daily_returns_list, cash_value = daily_returns(all_data, num_stocks, initial_portfolio_value)

  
    # calculate change in S&P 
    spy_change = []
    spy_cash = []
    #for each day 
    for i in range(1, len(spy)-1):
        # get the S&P price
         next_day_price = spy.iloc[i]["Open"]

         # get the new value of the S&P position
         next_day_value = num_spy_shares * next_day_price
         spy_cash.append(next_day_value)
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
        plt.title("Total % change in Portfolio Value vs S&P 500")
        plt.legend()
        plt.show()
        plot_returns(spy.index, daily_returns_list, daily_return_plot_start_idx)
        plot_cash_value(cash_value,spy_cash,spy_indicies_to_plot)

    pos_days, neg_days = return_data(daily_returns_list)
    print(f"Number of positive gain days: {pos_days} days")
    print(f"Number of negative gain days: {neg_days} days")

    return spy_change, percent_chage, spy.index, latest_date, daily_returns_list

def div_performance(stocks, weights, init_invest, recurring_deposit, recurring_rate, start_prices, start_ann_div_per_share, div_freq, div_cagr, stock_cagr, per,forcast_length_in_years):
    init_shares = []
    forcast_length_in_months = 12*forcast_length_in_years
    dataframes = []
    for i in range(len(stocks)):
        init_shares_stock = (init_invest * weights[i]) / start_prices[i]
        init_shares.append(init_shares_stock) 
        d = {
                "year" : 1, 
                "month" : 1, 
                "deposit" : init_invest*weights[i],
                'price' : start_prices[i],
                "purchased shares" : init_shares[i],
                'dividend' : 0,
                'dividend shares' : 0 ,
                'Cumlative Shares' : init_shares[i],
                'Ann. Div/Share' : start_ann_div_per_share[i],
                'Value of Shares' : init_shares[i] * start_prices[i]
            }
        dataframes.append(pd.DataFrame(d, index=[0]))
        dataframes[-1].name = stocks[i]
    totals = {
        "year" : 1, 
        "month" : 1, 
        "deposit" : init_invest,
        "purchased shares" : sum(init_shares),
        'dividend' : 0,
        'dividend shares' : 0 ,
        'Cumlative Shares' : sum(init_shares),
        'Ann. Div/Share' : sum(start_ann_div_per_share),
        'Value of Shares' : np.dot(init_shares, start_prices) 
    }
    fidx = 0
    for frame in dataframes:
        for i in range(1, forcast_length_in_months):
            # determine the year value
            if frame.iloc[i-1]['month'] == 12: 
                current_year = frame.iloc[i-1]['year'] + 1
            else:
                current_year = frame.iloc[i-1]['year']

            # determine the month 
            if frame.iloc[i-1]['month'] != 12: 
                current_month = frame.iloc[i-1]['month'] + 1
            else:
                current_month = 1

            # Determine Deposit 
            if np.mod(current_month, 12/recurring_rate) == 0: 
                # we are in a deposit month 
                deposit = recurring_deposit * weights[fidx]
            else: 
                deposit = 0 
            
            # determine the price 
            price = frame.iloc[i-1]['price'] + frame.iloc[i-1]['price'] * ((stock_cagr[fidx]/100)/12)

            purchased_shares = deposit / price 

            if np.mod(current_month, 12/div_freq[fidx]) == 0: 
                div = frame.iloc[i-1]['Cumlative Shares'] * frame.iloc[i-1]['Ann. Div/Share']/(div_freq[fidx])
            else: 
                div = 0 

            div_shares = div/price 

            cum_shares = div_shares + purchased_shares + frame.iloc[i-1]['Cumlative Shares']

            if frame.iloc[i-1]['month'] == 12: 
                #update div
                new_div = frame.iloc[i-1]['Ann. Div/Share'] + frame.iloc[i-1]['Ann. Div/Share'] * (div_cagr[fidx]/100)
            else: 
                new_div = frame.iloc[i-1]['Ann. Div/Share']

            value = cum_shares * price

            row = [current_year, current_month, deposit, price, purchased_shares, div, div_shares, cum_shares, new_div, value]

            frame.loc[len(frame)] = row
        fidx = fidx + 1
    
    totals = pd.DataFrame(totals, index=[0])
    for r in range(1, len( dataframes[0])):
    
     
        deposit = 0
        purchased = 0
        divs = 0
        div_shares = 0
        shares = 0
        divPerShare_ann = 0
        value = 0

        if frame.iloc[r-1]['month'] == 12: 
                current_year = frame.iloc[r-1]['year'] + 1
        else:
                current_year = frame.iloc[r-1]['year']

            # determine the month 
        if frame.iloc[i-1]['month'] != 12: 
                current_month = frame.iloc[r-1]['month'] + 1
        else:
                current_month = 1
        
        for frame in dataframes:
            deposit = deposit + frame.iloc[r]['deposit']
            purchased = purchased + frame.iloc[r]['purchased shares']
            divs = divs + frame.iloc[r]['dividend']
            div_shares = div_shares + frame.iloc[r]['dividend shares']
            shares = shares + frame.iloc[r]['Cumlative Shares']
            divPerShare_ann = divPerShare_ann + frame.iloc[r]['Ann. Div/Share']
            value = value + frame.iloc[r]['Value of Shares']
        row = [current_year, current_month, deposit, purchased, divs, div_shares, shares, divPerShare_ann, value]
        totals.loc[len(totals)] = row

    return totals, dataframes


def show_sector_exposure(stocks, weights):
    sect_totals = {}
    for t in stocks:
        idx = stocks.index(t) 
        obj = yf.Ticker(t)
        d = obj.get_info()
        ty = d['quoteType']

        # if the ticker is an ETF or mutual fund, we need to grab the weights of that security
        if ty == 'MUTUALFUND' or ty == 'ETF':
            # grab the sector weights in 
            sectWeights = d['sectorWeightings']
            sw = {}
            for dic in sectWeights: 
                k = list(dic.keys())
                sw[k[0]] = dic[k[0]]

            # rename all to be same as other dictionary
            # unfortunately the sectors are spelled differently with different case and underscores
            # need to remove and spell the same as the equity security types in order to match 

            temp = {}
            for s in sw.keys():
                if s == 'realestate': 
                    temp["Real Estate"] = sw[s]

                elif s == 'consumer_cyclical': 
                    temp['Consumer Cyclical'] = sw[s] 

                elif s == 'basic_materials': 
                    temp["Basic Materials"] = sw[s] 

                elif s == 'consumer_defensive': 
                    temp['Consumer Defensive'] = sw[s] 

                elif s == 'technology': 
                    temp['Technology'] = sw[s] 

                elif s == 'communication_services': 
                    temp['Communication Services']  = sw[s]

                elif s == 'financial_services': 
                    temp['Financial Services'] = sw[s] 
                    
                elif s == 'utilities': 
                    temp['Utilities'] = sw[s]

                elif s == 'industrials': 
                    temp['Industrials'] = sw[s] 

                elif s == 'energy': 
                    temp['Energy'] = sw[s] 

                else: 
                    temp['Health Care'] = sw[s]

            # loop through temp dictionary and check if the sector has been added to the sector values yet
            for key in temp.keys(): 
                # if there is not already a sector in the dictionary, add it 
                if key not in sect_totals.keys(): 
                    sect_totals[key] = weights[idx]
                # else add it to the current value
                else: 
                    sect_totals[key] = sect_totals[key] + weights[idx]
        else:
            # for equity security types (stocks)
            # grab the sector 
            sect = d['sector']

            # check to see if it is in the dictionary
            if sect not in sect_totals.keys(): 
                sect_totals[sect] = weights[idx]
            else: 
                sect_totals[sect] = sect_totals[sect] + weights[idx]
        

    # display 
    plt.pie(sect_totals.values(), labels=sect_totals.keys())

    my_circle = plt.Circle((0,0), 0.7, color='white')
    p = plt.gca()
    p.add_artist((my_circle))
    plt.show()



    