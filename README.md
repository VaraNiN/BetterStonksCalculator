# Better Stonks Calculator 📈

A realistic investment/retirement calculator that simulates portfolio growth using historical market data, including real returns, inflation, dividends, and taxes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VaraNiN/BetterStonksCalculator/blob/main/BetterStonksCalculator.ipynb)

## ⚠️ WARNING / WARNUNG

**This is amateur code and the author has no formal financial training!**

**Das ist amateur CODE und der Autor hat keine formale finanzielle Ausbildung!**

- Use at your own risk! / Benutzen auf eigene Gefahr!
- DO NOT use this code to inform real investment decisions.
- Verwenden Sie diesen Code NICHT, um echte Anlageentscheidungen zu treffen.
- No financial advice is given or implied.
- Es wird keine Finanzberatung gegeben oder angedeutet.
- The author is NOT responsible for any losses incurred by using this code.
- Der Autor ist NICHT verantwortlich für Verluste, die durch die Verwendung dieses Codes entstehen.

## 🚀 Quick Start (Google Colab - No Installation Required!)

**Click the "Open in Colab" badge above** to run this calculator directly in your browser without installing anything!

Just:
1. Click the badge
2. Wait for Colab to load
3. Run all cells (Runtime → Run all)
4. Adjust the parameters and re-run to see different scenarios

## 📊 What Does This Calculator Do?

This tool simulates long-term investment scenarios with:

- **Real historical stock market data** (from Yahoo Finance)
- **Actual inflation data** (US historical inflation rates)
- **Realistic distributions** of returns (not just simple averages)
- **Tax calculations** (capital gains tax)
- **Dividend reinvestment**
- **Inflation-adjusted contributions** (salary increases with inflation)
- **Monte Carlo simulations** (1000+ possible outcomes)

### Key Features

- ✅ Uses real historical data from any ticker symbol (SPY, VTI, Bitcoin, Gold, etc.)
- ✅ Accounts for dividends and inflation
- ✅ Simulates thousands of possible outcomes
- ✅ Shows both nominal and inflation-adjusted (real) returns
- ✅ Calculates IRR (Internal Rate of Return)
- ✅ Beautiful visualizations with histograms and distribution plots
- ✅ Fully customizable parameters

## 📈 Example Results

The calculator produces 6 plots:
1. **Portfolio value over time** - Shows growth trajectory with contributions
2. **Results summary** - Bar chart comparing nominal vs real values
3. **Returns distribution** - Shows how returns varied year-by-year
4. **Inflation distribution** - Shows actual inflation rates used
5. **Nominal outcome histogram** - Distribution of final portfolio values
6. **Real outcome histogram** - Distribution after accounting for inflation and taxes

## 🔧 Installation (For Local Use)

If you want to run this locally instead of using Google Colab:

### Requirements

- Python 3.8 or higher
- pip (Python package installer)

### Install Dependencies

```bash
pip install numpy numpy-financial scipy matplotlib yfinance tqdm pandas
```

### Download the Code

```bash
git clone https://github.com/VaraNiN/BetterStonksCalculator.git
cd BetterStonksCalculator
```

### Run the Calculator

```bash
python run.py
```

## ⚙️ Configuration

Edit the parameters at the top of `run.py`:

### Basic Parameters

```python
ticker_symbol = 'SPY'              # Stock/ETF ticker symbol
years = 25                         # Investment period
starting_sum = 0                   # Initial investment (€)
monthly_contribution = 500         # Monthly contribution (€)
capital_gains_tax = 27.5          # Tax rate (%)
simulation_iteration = 1000        # Number of simulations
increase_contribution_with_inflation = True  # Adjust contributions for inflation
```

### Return and Inflation Settings

```python
use_constant_returns = False       # Use fixed returns instead of historical
average_return = 13.06            # Expected annual return (%)
average_inflation = 2.56          # Expected annual inflation (%)
scramble_for_sample = False       # Randomize historical data vs use recent years
```

### Available Ticker Symbols

Some popular options:
- `SPY` - S&P 500 ETF (since 1993, includes dividends)
- `^GSPC` - S&P 500 Index (98 years, NO DIVIDENDS - not recommended)
- `VTI` - Total US Market ETF
- `EFA` - Developed Markets (since 2001)
- `EEM` - Emerging Markets (since 2003)
- `GC=F` - Gold futures (since 1974)
- `BTC-USD` - Bitcoin (since 2014)

## 📁 Data Files

The calculator automatically:
- Downloads historical stock data from Yahoo Finance
- Saves data to `data/` folder for faster subsequent runs
- Uses US inflation data from `data/inflation.csv`

## 📊 Understanding the Output

### Terminal Output

```
FINAL RESULTS FOR SAMPLE SIMULATION AFTER 25 YEARS:
Total nominal contributions: €150,000
Final nominal value: €456,789
Final real value after tax: €289,123

ANNUALIZED RETURNS:
Nominal IRR: 8.45%
Real IRR: 6.12%
Real IRR after tax: 5.34%
```

### Plot Interpretation

- **Blue lines**: Nominal (not adjusted for inflation)
- **Red lines**: Real (inflation-adjusted)
- **Dotted lines**: Constant return comparison
- **Black dashed line**: Your contributions
- **Histograms**: Show distribution of 1000+ simulated outcomes

## 🤝 Contributing

Found a bug or have a feature request? 
- Open an issue on GitHub
- Submit a pull request
- Contact the author

## 📝 Data Sources

- **Stock data**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance` package
- **Inflation data**: [US Inflation Calculator](https://www.usinflationcalculator.com/inflation/historical-inflation-rates/)

## 📜 License

This project is provided as-is for educational purposes. See the WARNING section above.

## 🙏 Acknowledgments

This calculator uses:
- NumPy for numerical computations
- Pandas for data handling
- Matplotlib for visualizations
- yfinance for market data
- SciPy for statistical distributions

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Remember**: Past performance does not guarantee future results. This is a simulation tool for educational purposes only!
