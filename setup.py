from distutils.core import setup

setup(name='fund performance comparison',
      version='3.0',
      description='Compare fund preformance to S&P 500 ETF',
      author='Brian Egolf',
      author_email='egolfbr@miamioh.edu',
      url='github.com',
      py_modules=["fund_performance"],
      install_requires=["yfinance", "numpy", "matplotlib"]
     )
