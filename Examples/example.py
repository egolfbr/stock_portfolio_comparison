from fund_performance import performance

stocks = ["NVDA", "INTC", "PSX", "QCOM", "FSDIX", "SPY", "AAPL"]

weights = [0.1442, 0.0867, 0.2779, 0.0634, 0.2239, 0.1710, 0.0329]

cash = 5000 

date = '2020-01-01'

spy_changes, portfolio_chages, dates, startdate, daily_return_list = performance(stocks, weights, cash, date, plot=True)