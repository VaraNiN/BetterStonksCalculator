import numpy as np
import numpy_financial as npf
import scipy.stats as stats
import matplotlib.pyplot as plt

###############################
# !!! WARNING !!!
# This is VIBE-CODED, amateur SPAGHETTI-CODE!
# Use at your own risk!
# DO NOT use to inform real investment decisions.
# No financial advice is given or implied.
# The author is NOT responsible for any losses incurred by using this code.
###############################



#####################################
##### Set basic parameters here #####
#####################################

years = 30 # Investment period
starting_sum = 0 # In €
monthly_contribution = 500  # In €
average_return = 7 # Annually, in percent
average_inflation = 2 # Annually, in percent
capital_gains_tax = 27.5 # In percent
increase_contribution_with_inflation = False # Simulate salary (and thus contribution) increasing with inflation?


###############################
##### Advanced parameters #####
###############################

minimum_return = -20    # Daily, in percent (20% is circuit breaker level 3 at NYSE)
maximum_return = 12     # Daily, in percent (11.58% was largest post WW2 daily increase of S&P 500 (13th Oct 2008))
freedom_return = 10   # How far the returns can deviate from the average (lower = more variance)

minimum_inflation = 0 # Yearly, in percent
maximum_inflation = 15 # Yearly, in percent
freedom_inflation = 1 # How far the returns can deviate from the average (lower = more variance)

force_exact_return = True   # (Recommended) Forces the geometric mean of the sampled returns to exactly equal average_return
force_exact_inflation = True # (Recommended) Forces the geometric mean of the sampled inflation to exactly equal average_inflation
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
    batch_size = int(n_samples * 1.5)  # oversample to ensure enough after filtering
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
    """Generate yearly returns using Student's t-distribution"""
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
    """Generate yearly inflation rates using Student's t-distribution"""
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

def run_investment_simulation(use_distribution_based_returns=False):
    """Run the complete investment simulation (yearly steps)"""
    # Generate random returns and inflation for each year
    if use_distribution_based_returns:
        daily_returns = generate_returns(years, bank_days_per_year)
        yearly_inflation = generate_inflation(years, 1)
    else:
        daily_returns = np.full(years*bank_days_per_year, (1 + average_return / 100)**(1/bank_days_per_year))
        yearly_inflation = np.full(years, (1 + average_inflation / 100))


    # Track inflation-adjusted yearly contributions
    current_contribution = monthly_contribution
    nominal_values = [starting_sum]
    real_values = [starting_sum]
    nominal_contributions = [starting_sum]
    real_contributions = [starting_sum] 
    inflation_factor_yearly = 1.0
    inflation_factor_daily = 1.0
    inflation_data = [inflation_factor_yearly]
    yearly_returns = np.zeros(years)    

    # Run simulation year by year
    day_index = 0
    for year in range(years):
        for month in range(12):
            nominal_values[-1] += current_contribution # Add monthly contribution
            nominal_contributions.append(current_contribution)
            real_contributions.append(nominal_contributions[-1] / inflation_factor_yearly)
            for day in range(bank_days_per_month):
                # Apply correct daily return to current portfolio
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





##################################
##### Simulation & Plotting ######
##################################

# Run the investment simulation
print("Running Calculator...")
print(f"Initial investment: €{starting_sum:,}")
print(f"Yearly contribution: €{monthly_contribution * 12:,}")
print(f"Investment period: {years} years")
print(f"Expected return: {average_return}% (range: {minimum_return}% to {maximum_return}%)")
print(f"Expected inflation: {average_inflation}% (range: {minimum_inflation}% to {maximum_inflation}%)")
print(f"Force exact averages: Returns={force_exact_return}, Inflation={force_exact_inflation}")
print("-" * 60)

# Run simulation
nominal_values, real_values, yearly_returns, yearly_inflation, nominal_contributions, real_contributions, inflation_data = run_investment_simulation(True)
# Also get constant interest case for comparison
const_nominal_values, const_real_values, _, _, _, _, _ = run_investment_simulation(False)

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

plt.figure(figsize=(15, 12))

# Portfolio value over time
plt.subplot(2, 2, 1)
plt.plot(years_axis, nominal_values, 'b-', linewidth=2, label='Nominal Value (Stochastic)')
plt.plot(years_axis, real_values, 'r-', linewidth=2, label='Real Value (Stochastic)')
plt.plot(years_axis, const_nominal_values, 'b:', linewidth=2, label='Nominal Value (Constant)')
plt.plot(years_axis, const_real_values, 'r:', linewidth=2, label='Real Value (Constant)')
plt.plot(years_axis, cum_nominal_contrib_daily, 'k--', linewidth=1, label='Total Contributions (Nominal)')
plt.plot(years_axis, cum_real_contrib_daily, 'm--', linewidth=1, label='Total Contributions (Real)')
plt.title('Portfolio Value Over Time')
plt.xlabel('Years')
plt.ylabel('Portfolio Value (€)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, years)
plt.xticks(np.arange(0, years+1, max(1, years//10)))

# Add cumulative inflation on a second y-axis
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(years_axis, (inflation_data-1)*100, 'c-', linewidth=2, label='Cumulative Inflation (%)')
ax2.set_ylabel('Cumulative Inflation (%)')
ax2.legend(loc='upper right')

# Final summary
plt.subplot(2, 2, 2)
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
bars = plt.bar(categories, values, color=colors, alpha=0.7)
for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'€{value:,.0f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.title('Final Results Summary')
plt.ylabel('Value (€)')

# Returns distribution
plt.subplot(2, 2, 3)
plt.hist((yearly_returns - 1) * 100, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Yearly Returns')
plt.xlabel('Yearly Return (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Inflation distribution
plt.subplot(2, 2, 4)
plt.hist((yearly_inflation - 1) * 100, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribution of Yearly Inflation')
plt.xlabel('Yearly Inflation (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final statistics
print(f"\nFINAL RESULTS AFTER {years} YEARS:")
print(f"Total contributions: €{total_contributions:,.0f}")
print(f"Final nominal value: €{final_nominal:,.0f}")
print(f"Nominal tax: €{nominal_tax:,.0f}")
print(f"Final nominal value after {capital_gains_tax}% tax: €{nominal_after_tax:,.0f}")

print(f"\nFinal real value: €{final_real:,.0f}")
print(f"Real tax: €{real_tax:,.0f}")
print(f"Final real value after after {capital_gains_tax}% tax: €{real_after_tax:,.0f}")

# Build cash flow arrays for IRR calculation
cash_flows = [-c for c in nominal_contributions]
cash_flows.append(final_nominal)
nominal_IRR = (1+npf.irr(cash_flows))**(12)-1

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


