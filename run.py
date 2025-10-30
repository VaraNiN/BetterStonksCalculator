import numpy as np
import numpy_financial as npf
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from tqdm import tqdm
import os
import pandas as pd

###############################
# !!! WARNING // WARNUNG !!!

# This is amateur CODE and the author has no formal financial training! 
# Das ist amateur CODE und der Autor hat keine formale finanzielle Ausbildung!

# Use at your own risk! 
# Benutzen auf eigene Gefahr!

# DO NOT use this code to inform real investment decisions. 
# Verwenden Sie diesen Code NICHT, um echte Anlageentscheidungen zu treffen.

# No financial advice is given or implied. 
# Es wird keine Finanzberatung gegeben oder angedeutet.

# The author is NOT responsible for any losses incurred by using this code.
# Der Autor ist NICHT verantwortlich für Verluste, die durch die Verwendung dieses Codes entstehen.
###############################

# Source inflation data: https://www.usinflationcalculator.com/inflation/historical-inflation-rates/
# Source for stock data: Yahoo Finance via yfinance package
# Up to date source code: https://github.com/VaraNiN/BetterStonksCalculator


#####################################
##### Set basic parameters here #####
#####################################

ticker_symbol = 'SPY'  # Ticker symbol from yfinance (any works, here are some examples):
#   ^GSPC for S&P 500 index: 98 years of data, but NO DIVIDENDS INCLUDED (and thus wrong total returns!!)
#   SPY for S&P 500 ETF with dividends: Since 1993
#   EFA for Developed Markets since 2001
#   EEM for Emerging Markets since 2003
#   GC=F for Gold future contracts since 1974
#   BTC-USD for Bitcoin since 2014

years = 25 # Investment period
starting_sum = 0 # In $
monthly_contribution = 500  # In $
capital_gains_tax = 27.5 # In percent
simulation_iteration = 1000 # Number of simulation iterations (reduce for faster results, increase for smoother results)
increase_contribution_with_inflation = True # Simulate salary (and thus contribution) increasing with inflation?

use_constant_returns = False # If True, uses constant returns/inflation, otherwise uses synthetic or historical data
average_return = 13.06 # Annually, in percent; gets used for constant returns and as target for synthetic data
average_inflation = 2.56 # Annually, in percent; gets used for constant inflation and as target for synthetic data

# If True, scrambles historical data for sample simulation, otherwise uses the last real years. 
# Always scrambles for histogram simulations.
scramble_for_sample = False


##################################################
##### Advanced parameters for synthetic data #####
##################################################

use_synthetic_data = False 
# If True, uses synthetic data based on the student_t distribution (this is a VERY simplified model!)
# If False, uses data from ticker_symbol

minimum_return = -20    # Daily, in percent (20% is circuit breaker level 3 at NYSE)
maximum_return = 12     # Daily, in percent (11.58% was largest post WW2 daily increase of S&P 500 (13th Oct 2008))
freedom_return = 10   # How far the returns can deviate from the average (lower = more variance)

minimum_inflation = 0 # Yearly, in percent
maximum_inflation = 15 # Yearly, in percent
freedom_inflation = 1 # How far the returns can deviate from the average (lower = more variance)

force_exact_return = True   # (Highly Recommended) Forces the geometric mean of the sampled returns to exactly equal average_return
force_exact_inflation = True # (Highly Recommended) Forces the geometric mean of the sampled inflation to exactly equal average_inflation
fixed_seed = 0          # If non-zero, uses this seed for random number generation (for reproducibility)

bank_days_per_month = 21  # Trading days in a month (NYSE standard)


#########################################################
###### No more user input required below this line ######
#########################################################










#################################
##### Function Definitions ######
#################################

bank_days_per_year = 12 * bank_days_per_month 
freedom_return *= bank_days_per_year

def sample_student_t(years, samples_per_year=252, df=1, ARR=2, minimum=0, maximum=10, force_exact=False, fixed_seed=42):
    """
    Vectorized sampling from truncated Student's t-distribution
    df: degrees of freedom (how much variance there is)
    ARR: targeted annualized rate of return
    minimum: minimum value (truncation)
    maximum: maximum value (truncation)
    force_exact: if True, forces the geometric mean of the samples to exactly equal ARR
    fixed_seed: if non-zero, uses it as seed for reproducibility
    """
    if fixed_seed:
        np.random.seed(fixed_seed)
    n_samples = years * samples_per_year
    batch_size = int(n_samples * 2.0)  # oversample to ensure enough after filtering
    samples = []
    while len(samples) < n_samples:
        raw = stats.t.rvs(df, loc=(1+ARR)**(1/samples_per_year)-1, size=batch_size)
        filtered = raw[(filtered := (raw >= minimum) & (raw <= maximum))]
        samples.extend(filtered.tolist())
    samples = np.array(samples[:n_samples])
    results = 1 + samples / 100.
    if force_exact:
        results = results * ((1 + ARR/100.)**(1/samples_per_year) / (stats.gmean(results)))
    return results

def generate_returns(years, samples_per_year):
    """Generate returns using Student's t-distribution"""
    return sample_student_t(
        years=years,
        samples_per_year=samples_per_year,
        df=freedom_return,
        ARR=average_return,
        minimum=minimum_return,
        maximum=maximum_return,
        force_exact=force_exact_return,
        fixed_seed=fixed_seed
    )

def generate_inflation(years, samples_per_year):
    """Generate inflation rates using Student's t-distribution"""
    return sample_student_t(
        years=years,
        samples_per_year=samples_per_year,
        df=freedom_inflation,
        ARR=average_inflation,
        minimum=minimum_inflation,
        maximum=maximum_inflation,
        force_exact=force_exact_inflation,
        fixed_seed=fixed_seed
    )

def run_investment_simulation(use_constant_returns, use_synthetic_data, ticker_years = None, scramble_for_sample = True, inflation_csv_path="data/inflation.csv"):
    """
    Run the complete investment simulation (yearly steps).
    - If use_constant_returns is True: use constant returns/inflation.
    - If use_constant_returns is False and use_synthetic_data is True: use Student's t-distribution for returns/inflation.
    - If use_constant_returns is False and use_synthetic_data is False: use real historical ticker data (scrambled years) and matching real inflation data from CSV.
    """
    if use_constant_returns:
        daily_returns = np.full(years*bank_days_per_year, (1 + average_return / 100)**(1/bank_days_per_year))
        yearly_inflation = np.full(years, (1 + average_inflation / 100))
    elif use_synthetic_data:
        daily_returns = generate_returns(years, bank_days_per_year)
        yearly_inflation = generate_inflation(years, 1)
    else:
        # Use real historical ticker data, scramble years
        if ticker_years is None:
            raise ValueError("ticker_years must be provided when using real data.")
        available_years = list(ticker_years.keys())
        if len(available_years) < years:
                print(f"Not enough data in {ticker_symbol}: {years} years required, only {len(available_years)} years available.")
                exit(1)
        if scramble_for_sample:
            chosen_years = np.random.choice(available_years, years, replace=False)
        else:
            chosen_years = sorted(available_years)[-years:]
        daily_returns = []
        for y in chosen_years:
            year_data = ticker_years[y]
            closes = year_data['closes']
            dividends = year_data['dividends']
            # Calculate daily returns including dividends: (Close[t] + Dividend[t]) / Close[t-1]
            returns = (closes[1:] + dividends[1:]) / closes[:-1]
            # Pad to bank_days_per_year if needed
            if len(returns) < bank_days_per_year:
                returns = np.pad(returns, (0, bank_days_per_year - len(returns)), 'edge')
            else:
                returns = returns[:bank_days_per_year]
            daily_returns.extend(returns)
        daily_returns = np.array(daily_returns)
        # Use matching real inflation data from CSV
        yearly_inflation = []
        if inflation_csv_path and os.path.exists(inflation_csv_path):
            try:
                infl_df = pd.read_csv(inflation_csv_path)
                infl_aves = pd.to_numeric(infl_df['Ave'].values, errors='coerce')
                year_to_infl = dict(zip(infl_df['Year'].values, infl_aves))
                for y in chosen_years:
                    infl = year_to_infl.get(y, np.nan)
                    if np.isnan(infl):
                        infl = average_inflation
                    yearly_inflation.append(1 + infl / 100.0)
            except Exception as e:
                print(f"Warning: Could not read inflation data from {inflation_csv_path}: {e}")
                print(f"Using average inflation of {average_inflation}% for all years.")
                yearly_inflation = [1 + average_inflation / 100.0 for _ in chosen_years]
        else:
            if inflation_csv_path:
                print(f"Warning: Inflation data file not found at {inflation_csv_path}")
            print(f"Using average inflation of {average_inflation}% for all years.")
            yearly_inflation = [1 + average_inflation / 100.0 for _ in chosen_years]
        yearly_inflation = np.array(yearly_inflation)


    # Tracking variables
    current_contribution = monthly_contribution
    nominal_values = [starting_sum]
    real_values = [starting_sum]
    nominal_contributions = [starting_sum]
    real_contributions = [starting_sum] 
    inflation_factor_yearly = 1.0
    inflation_factor_daily = 1.0
    inflation_data = [inflation_factor_yearly]
    yearly_returns = np.zeros(years)    

    # Run simulation
    day_index = 0
    for year in range(years):
        for month in range(12):
            nominal_values[-1] += current_contribution # Add monthly contribution
            nominal_contributions.append(current_contribution)
            real_contributions.append(nominal_contributions[-1] / inflation_factor_yearly)
            for day in range(bank_days_per_month):
                # Apply daily return to current portfolio
                inflation_factor_daily *= yearly_inflation[year]**(1/bank_days_per_year)
                nominal_values.append(nominal_values[-1] * daily_returns[day_index])
                real_values.append(nominal_values[-1] / inflation_factor_daily)
                inflation_data.append(inflation_factor_daily)
                day_index += 1
        inflation_factor_yearly *= yearly_inflation[year]
        yearly_returns[year] = np.prod(daily_returns[year*bank_days_per_year:(year+1)*bank_days_per_year])
        # Increase monthly contribution with inflation if enabled
        if increase_contribution_with_inflation:
            current_contribution = monthly_contribution * inflation_factor_yearly
    return (
        np.array(nominal_values),
        np.array(real_values),
        np.array(yearly_returns),
        np.array(yearly_inflation),
        np.array(nominal_contributions),
        np.array(real_contributions),
        np.array(inflation_data)
    )

def get_ticker_history(ticker_symbol, bank_days_per_year=252):
    """
    Loads daily closing prices and dividends for a ticker from local CSV if available, otherwise downloads from Yahoo Finance and saves to CSV.
    Splits the data into full years, discarding years with less than (bank_days_per_year - 10) entries.
    Returns a dict: {year: {'closes': np.array, 'dividends': np.array}}
    """
    # Create data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a safe filename from the ticker symbol
    safe_ticker = ticker_symbol.replace('^', '').replace('/', '_')
    csv_path = os.path.join(data_dir, f'{safe_ticker}_history.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading {ticker_symbol} history from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    else:
        print(f"Downloading {ticker_symbol} history from Yahoo Finance...")
        data = yf.download(ticker_symbol, start='1900-01-01', progress=False, actions=True)
        # Include Close and Dividends columns
        if 'Dividends' in data.columns:
            df = data[['Close', 'Dividends']].reset_index()
        else:
            df = data[['Close']].reset_index()
            df['Dividends'] = 0.0
        df.to_csv(csv_path, index=False)
        print(f"Saved {ticker_symbol} history to {csv_path}.")
    
    # Ensure Dividends column exists
    if 'Dividends' not in df.columns:
        df['Dividends'] = 0.0
    
    df['Year'] = df['Date'].dt.year
    years_dict = {}
    for year, group in df.groupby('Year'):
        closes = group['Close'].values
        dividends = group['Dividends'].values
        # Convert to float, handling any potential string values
        closes = np.array([float(np.asarray(x).item()) if not pd.isna(x) else np.nan for x in closes])
        dividends = np.array([float(np.asarray(x).item()) if not pd.isna(x) else 0.0 for x in dividends])
        # Remove NaN values (keep indices synchronized)
        valid_indices = ~np.isnan(closes)
        closes = closes[valid_indices]
        dividends = dividends[valid_indices]
        if len(closes) >= (bank_days_per_year - 10):
            years_dict[year] = {'closes': closes, 'dividends': dividends}
    return years_dict

def custom_log_formatter(x, pos):
    """
    Formats the tick label 'x' (the tick value) into a human-readable string.
    Handles both regular numbers and scientific notation from matplotlib.
    """
    # Convert to float to handle any potential type issues
    try:
        x = float(x)
    except (ValueError, TypeError):
        return str(x)
    
    if abs(x) >= 1e5 and abs(x) < 1e9:
        if abs(x/1e5 - round(x/1e5)) < 0.01:
            # It's close to a whole number
            return f"{x/1e6:,.1f} Mio."
        else:
            return f"{int(x):,}"
    elif abs(x) >= 1e8 and abs(x) < 1e12:
        return f"{x/1e9:,.1f} Mrd."
    elif abs(x) >= 1:
        return f"{int(x):,}"
    else:
        # For values less than 1
        return f'{x:g}'




##################################
##### Simulation & Plotting ######
##################################

# Run the investment simulation
print("Running Calculator...")
print(f"Initial investment: ${starting_sum:,}")
print(f"Yearly contribution: ${monthly_contribution * 12:,}")
print(f"Investment period: {years} years")
if use_synthetic_data:
    print("Using synthetic data based on Student's t-distribution.")
    print(f"Expected return: {average_return}% (range: {minimum_return}% to {maximum_return}%)")
    print(f"Expected inflation: {average_inflation}% (range: {minimum_inflation}% to {maximum_inflation}%)")
    print(f"Force exact averages: Returns={force_exact_return}, Inflation={force_exact_inflation}")
else:
    print(f"Using historical {ticker_symbol} data.")
print("-" * 60)

# Run simulation
if not use_synthetic_data:
    ticker_years = get_ticker_history(ticker_symbol)
nominal_values, real_values, yearly_returns, yearly_inflation, nominal_contributions, real_contributions, inflation_data = run_investment_simulation(use_constant_returns, use_synthetic_data, ticker_years=ticker_years, scramble_for_sample=scramble_for_sample)
# Also get constant interest case for comparison
const_nominal_values, const_real_values, _, _, _, _, _ = run_investment_simulation(True, False)

# Calculate cumulative contributions for plotting
cum_nominal_contrib = np.cumsum(nominal_contributions)
cum_real_contrib = np.cumsum(real_contributions)

# Build x-axis for plotting (daily resolution)
n_days = len(nominal_values)
days_axis = np.arange(n_days)
years_axis = days_axis / (12 * bank_days_per_month)

# Interpolate cumulative contributions to daily resolution
monthly_indices = np.linspace(0, n_days-1, len(cum_nominal_contrib))
cum_nominal_contrib_daily = np.interp(days_axis, monthly_indices, cum_nominal_contrib)
cum_real_contrib_daily = np.interp(days_axis, monthly_indices, cum_real_contrib)

# Run multiple simulations for histogram analysis
nominal_results = []
real_after_tax_results = []
for i in tqdm(range(simulation_iteration), desc="Simulating", ncols=80):
    sim_nominal_values, sim_real_values, _, sim_yearly_inflation, sim_nominal_contributions, _, _ = run_investment_simulation(False, use_synthetic_data, ticker_years=ticker_years if not use_synthetic_data else None)
    sim_total_contributions = np.cumsum(sim_nominal_contributions)[-1]
    sim_final_nominal = sim_nominal_values[-1]
    sim_final_real = sim_real_values[-1]
    sim_nominal_tax = (sim_final_nominal - sim_total_contributions) * (capital_gains_tax / 100)
    sim_real_tax = sim_nominal_tax / np.prod(sim_yearly_inflation)
    sim_real_after_tax = sim_final_real - sim_real_tax
    nominal_results.append(sim_final_nominal)
    real_after_tax_results.append(sim_real_after_tax)

plt.figure(figsize=(18, 12))

# Add main title at the top of the figure
if use_synthetic_data:
    main_title = f"Synthetic Data - {years} Year Investment Simulation"
else:
    main_title = f"{ticker_symbol} - {years} Year Investment Simulation"
plt.suptitle(main_title, fontsize=20, fontweight='bold', y=0.995)

# Portfolio value over time
ax = plt.subplot(3, 2, 1)
if use_synthetic_data:
    string = "Stochastic"
elif scramble_for_sample:
    string = "Scrambled Historical"
else:
    string = "Historical"
ax.plot(years_axis, nominal_values, 'b-', linewidth=2, label='Nominal Value (%s)' % string)
ax.plot(years_axis, real_values, 'r-', linewidth=2, label='Real Value (%s)' % string)
ax.plot(years_axis, const_nominal_values, 'b:', linewidth=2, label=f'Nominal Value (Constant {average_return:.2f}%p.a. returns)')
ax.plot(years_axis, const_real_values, 'r:', linewidth=2, label=f'Real Value (Constant {average_inflation:.2f}%p.a. inflation)')
ax.plot(years_axis, cum_nominal_contrib_daily, 'k--', linewidth=1, label='Total Contributions (Nominal)')
ax.plot(years_axis, cum_real_contrib_daily, 'm--', linewidth=1, label='Total Contributions (Real)')
ax.set_title('Sample Simulation: Portfolio Value Over Time')
ax.set_xlabel('Years')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, years)
ax.set_xticks(np.arange(0, years+1, max(1, years//10)))
formatter = FuncFormatter(custom_log_formatter)
ax.yaxis.set_major_formatter(formatter)

# Add cumulative inflation on a second y-axis
ax2 = ax.twinx()
ax2.plot(years_axis, (inflation_data-1)*100, 'c-', linewidth=2, label='Cumulative Inflation (%)')
ax2.set_ylabel('Cumulative Inflation (%)')
ax2.legend(loc='lower right')

# Final summary
ax = plt.subplot(3, 2, 2)
final_nominal = nominal_values[-1]
final_real = real_values[-1]
total_contributions = cum_nominal_contrib[-1]
final_real_contribution_total = cum_real_contrib[-1]


nominal_tax = (final_nominal - total_contributions) * (capital_gains_tax / 100)
nominal_after_tax = final_nominal - nominal_tax

real_tax = nominal_tax / np.prod(yearly_inflation)
real_after_tax = final_real - real_tax

categories = ['Total Nominal\nContributions', 'Total Real\nContributions', 'Final Nominal\nValue', 'Final Real\nValue', 'Final Real\nValue After Tax']
values = [total_contributions, final_real_contribution_total, final_nominal, final_real, real_after_tax]
colors = ['gray', 'magenta', 'blue', 'red', 'purple']
bars = ax.bar(categories, values, color=colors, alpha=0.7)
ax.set_ylim(0, max(values)*1.1)  # Make the plot taller to avoid clipping
formatter = FuncFormatter(custom_log_formatter)
ax.yaxis.set_major_formatter(formatter)
for bar, raw_value in zip(bars, [total_contributions, final_real_contribution_total, final_nominal, final_real, real_after_tax]):
    height = bar.get_height()
    if raw_value >= 1e9:
        label = f'{raw_value/1e6:.2f} Mio. $'
    else:
        label = f'{raw_value:,.0f} $'
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            label, ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.title('Sample Simulation: Results Summary')
plt.ylabel('Value ($)')

# Returns distribution
plt.subplot(3, 2, 3)
plt.hist((yearly_returns - 1) * 100, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.title('Sample Simulation: Distribution of Yearly Returns')
plt.xlabel('Yearly Return (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Inflation distribution
plt.subplot(3, 2, 4)
plt.hist((yearly_inflation - 1) * 100, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.title('Sample Simulation: Distribution of Yearly Inflation')
plt.xlabel('Yearly Inflation (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# New subplot: Histogram of final nominal returns
ax = plt.subplot(3, 2, 5)
nominal_results_mio = np.array(nominal_results) #/ 1e6
min_val = max(nominal_results_mio.min(), 1e-2)
max_val = nominal_results_mio.max()
bins = np.logspace(np.log10(min_val), np.log10(max_val), 21)
hist, bin_edges = np.histogram(nominal_results_mio, bins=bins)
ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7, color='blue', edgecolor='black')
ax.set_xscale('log')
log_min = int(np.floor(np.log10(min_val)))
log_max = int(np.ceil(np.log10(max_val)))
ticks = []
for i in range(log_min, log_max + 1):
    ticks.extend([10**i, 3 * 10**i])#, 7 * 10**i])
ticks = [t for t in ticks if min_val <= t <= max_val]
ax.set_xticks(ticks)
formatter = FuncFormatter(custom_log_formatter)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: ''))
median_nominal = np.median(nominal_results) #/ 1e6
ax.axvline(median_nominal, color='black', linestyle='--', linewidth=2, label=f'Median: {median_nominal:,.0f} $')
ax.set_title(f'Histogram of Nominal Portfolio Value ({simulation_iteration} Simulations)')
ax.set_xlabel('Final Nominal Value ($)')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

# New subplot: Histogram of final real returns after tax
ax = plt.subplot(3, 2, 6)
real_after_tax_mio = np.array(real_after_tax_results) #/ 1e6
min_val = max(real_after_tax_mio.min(), 1e-2)
max_val = real_after_tax_mio.max()
bins = np.logspace(np.log10(min_val), np.log10(max_val), 21)
hist, bin_edges = np.histogram(real_after_tax_mio, bins=bins)
ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', alpha=0.7, color='red', edgecolor='black')
ax.set_xscale('log')
log_min = int(np.floor(np.log10(min_val)))
log_max = int(np.ceil(np.log10(max_val)))
ticks = []
for i in range(log_min, log_max + 1):
    ticks.extend([10**i, 3 * 10**i])#, 7 * 10**i])
ticks = [t for t in ticks if min_val <= t <= max_val]
ax.set_xticks(ticks)
formatter = FuncFormatter(custom_log_formatter)
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: ''))
median_real_after_tax = np.median(real_after_tax_results) #/ 1e6
ax.axvline(median_real_after_tax, color='black', linestyle='--', linewidth=2, label=f'Median: {median_real_after_tax:,.0f} $')
ax.set_title(f'Histogram of Real Portfolio Value After Tax ({simulation_iteration} Simulations)')
ax.set_xlabel('Final Real Value After Tax ($)')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=300)
plt.show()


# Print final statistics
print(f"\nFINAL RESULTS FOR SAMPLE SIMULATION AFTER {years} YEARS:")
print(f"Total nominal contributions: ${total_contributions:,.0f}")
print(f"Final nominal value: ${final_nominal:,.0f}")
print(f"Nominal tax: ${nominal_tax:,.0f}")
print(f"Final nominal value after {capital_gains_tax}% tax: ${nominal_after_tax:,.0f}")

print(f"\nTotal real contributions: ${final_real_contribution_total:,.0f}")
print(f"Final real value: ${final_real:,.0f}")
print(f"Real tax: ${real_tax:,.0f}")
print(f"Final real value after {capital_gains_tax}% tax: ${real_after_tax:,.0f}")

# Build cash flow arrays for IRR calculation
cash_flows = [-c for c in nominal_contributions]
cash_flows.append(final_nominal)
nominal_IRR = (1+npf.irr(cash_flows))**(12)-1

cash_flows = [-c for c in real_contributions]
cash_flows[-1] = final_real
real_IRR = (1+npf.irr(cash_flows))**(12)-1

cash_flows[-1] = real_after_tax
real_IRR_after_tax = (1+npf.irr(cash_flows))**(12)-1

print("\nANNUALIZED RETURNS:")
print(f"Nominal IRR: {nominal_IRR * 100:.2f}%")
print(f"Real IRR: {real_IRR * 100:.2f}%")
print(f"Real IRR after tax: {real_IRR_after_tax * 100:.2f}%")
print(f"Actual average yearly return: {(stats.gmean(yearly_returns)-1) * 100:.2f}%")
print(f"Actual average yearly inflation: {(stats.gmean(yearly_inflation)-1) * 100:.2f}%")


