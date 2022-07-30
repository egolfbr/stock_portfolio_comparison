from distutils.core import setup

setup(name='Fund Performance Comparison Tool',
      version='3.9.7',
      description='Compare fund preformance to S&P 500 ETF',
      author='Brian Egolf',
      author_email='egolfbr@miamioh.edu',
      url='https://github.com/egolfbr/stock_portfolio_comparison',
      py_modules=["fund_performance"],
      install_requires=["yfinance", "numpy", "matplotlib"]
     )
