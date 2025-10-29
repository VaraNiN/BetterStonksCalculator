import numpy as np
import numpy_financial as npf
import scipy.stats as stats
import matplotlib.pyplot as plt

##### This is vibe-coded, amateur spaghetti-code. Use at your own risk! #####
##### Do NOT use to inform real investment decisions. #####
##### No financial advice is given or implied. #####
##### The author is not responsible for any losses incurred by using this code. #####



###############################
##### Set parameters here #####
###############################

years = 30 # Investment period
starting_sum = 0 # In €
monthly_contribution = 500  # In €
average_return = 7 # In percent
average_inflation = 2 # In percent
capital_gains_tax = 27.5 # In percent
increase_contribution_with_inflation = True # Simulate salary (and thus contribution) increasing with inflation?


###############################
##### Advanced parameters #####
###############################

# Instead of using the usual constant average return and inflation rates,
# do you want to use a more realistic distribution-based approach? (Recommended)
use_distribution_based_returns = True

minimum_return = -50 # In percent
maximum_return = 40 # In percent
freedom_return = 0.1    # How far the returns can deviate from the average (lower = more variance)

minimum_inflation = -0.5 # In percent
maximum_inflation = 15 # In percent
freedom_inflation = 0.5 # How far the returns can deviate from the average (lower = more variance)

force_exact_return = True   # (Recommended) Forces the geometric mean of the sampled returns to exactly equal average_return
force_exact_inflation = True # (Recommended) Forces the geometric mean of the sampled inflation to exactly equal average_inflation
fixed_seed = 0          # If non-zero, uses this seed for random number generation (for reproducibility)


#########################################################
###### No more user input required below this line ######
#########################################################










#################################
##### Function Definitions ######
#################################

def sample_student_t(n_samples, df=2, loc=2, minimum=0, maximum=10, force_exact=False, fixed_seed=42):
    """
    Vectorized sampling from truncated Student's t-distribution
    df: degrees of freedom (how much variance there is
    loc: mean of the distribution
    minimum: minimum value (truncation)
    maximum: maximum value (truncation)
    force_exact: if True, forces the geometric mean of the samples to equal loc
    fixed_seed: if non-zero, uses it as seed for reproducibility
    """
    if fixed_seed:
        np.random.seed(fixed_seed)
    batch_size = int(n_samples * 1.5)  # oversample to ensure enough after filtering
    samples = []
    while len(samples) < n_samples:
        raw = stats.t.rvs(df, loc=loc, size=batch_size)
        filtered = raw[(filtered := (raw >= minimum) & (raw <= maximum))]
        samples.extend(filtered.tolist())
    samples = np.array(samples[:n_samples])
    results = 1 + samples / 100.
    if force_exact:
        results = results * ((1 + loc/100.) / (stats.gmean(results)))
    return results

def generate_returns(years):
    """Generate yearly returns using Student's t-distribution"""
    return sample_student_t(
        n_samples=years,
        df=freedom_return,
        loc=average_return,
        minimum=minimum_return,
        maximum=maximum_return,
        force_exact=force_exact_return,
        fixed_seed=fixed_seed
    )

def generate_inflation(years):
    """Generate yearly inflation rates using Student's t-distribution"""
    return sample_student_t(
        n_samples=years,
        df=freedom_inflation,
        loc=average_inflation,
        minimum=minimum_inflation,
        maximum=maximum_inflation,
        force_exact=force_exact_inflation,
        fixed_seed=fixed_seed
    )

def run_investment_simulation():
    """Run the complete investment simulation (yearly steps)"""
    # Generate random returns and inflation for each year
    if use_distribution_based_returns:
        yearly_returns = generate_returns(years)
        yearly_inflation = generate_inflation(years)
    else:
        yearly_returns = np.full(years, 1 + average_return / 100)
        yearly_inflation = np.full(years, 1 + average_inflation / 100)
    
    # Initialize tracking arrays
    portfolio_value = np.zeros(years + 1)
    real_value = np.zeros(years + 1)  # Inflation-adjusted value
    portfolio_value[0] = starting_sum
    real_value[0] = starting_sum
    
    # Cumulative inflation factor
    inflation_factor = 1.0
    
    # Track inflation-adjusted yearly contribution
    current_contribution = monthly_contribution * 12
    contributions = [starting_sum]  # Track all contributions
    
    # Run simulation year by year
    for year in range(years):
        # Apply yearly return to current portfolio
        portfolio_value[year + 1] = portfolio_value[year] * yearly_returns[year]
        
        # Add yearly contribution
        if increase_contribution_with_inflation:
            current_contribution *= yearly_inflation[year]
        portfolio_value[year + 1] += current_contribution
        contributions.append(current_contribution)
        
        # Update inflation factor
        inflation_factor *= yearly_inflation[year]

        # Calculate real (inflation-adjusted) value
        real_value[year + 1] = portfolio_value[year + 1] / inflation_factor
    
    total_contributions = np.sum(contributions)
    return portfolio_value, real_value, yearly_returns, yearly_inflation, total_contributions, contributions









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
portfolio_values, real_values, returns, inflation_rates, total_contributions, contributions = run_investment_simulation()

# Calculate real contributions before plotting
real_contributions = [contributions[0]]
real_factor = 1.0
for i in range(1, len(contributions)):
    real_factor *= inflation_rates[i-1] if i-1 < len(inflation_rates) else 1.0
    real_contributions.append(contributions[i] / real_factor)

# Plotting (no extra function)
years_axis = np.arange(len(portfolio_values))
plt.figure(figsize=(15, 12))

# Portfolio value over time
plt.subplot(2, 2, 1)
plt.plot(years_axis, portfolio_values, 'b-', linewidth=2, label='Nominal Value')
plt.plot(years_axis, real_values, 'r-', linewidth=2, label='Real Value (Inflation-Adjusted)')
plt.plot(years_axis, [total_contributions * (i / years) for i in years_axis], 'k--', linewidth=1, label='Total Contributions (Nominal)')
plt.plot(years_axis, np.cumsum(real_contributions), 'm--', linewidth=1, label='Total Contributions (Real)')
plt.title('Portfolio Value Over Time')
plt.xlabel('Years')
plt.ylabel('Portfolio Value (€)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Add cumulative inflation on a second y-axis
ax1 = plt.gca()

# Second y-axis for cumulative inflation
ax2 = ax1.twinx()
cumulative_inflation = np.cumprod(inflation_rates)
ax2.plot(years_axis[1:], cumulative_inflation, 'c-', linewidth=2, label='Cumulative Inflation')
ax2.set_ylabel('Cumulative Inflation Factor')
ax2.legend(loc='upper right')

# Final summary
plt.subplot(2, 2, 2)
final_nominal = portfolio_values[-1]
final_real = real_values[-1]
real_after_tax = final_real - (final_nominal - total_contributions) * (capital_gains_tax / 100)
final_real_contribution_total = np.sum(real_contributions)
categories = ['Total\nContributions', 'Final Real\nContributions', 'Final Nominal\nValue', 'Final Real\nValue', 'Final Real\nValue After Tax']
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
plt.hist((returns - 1) * 100, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Yearly Returns')
plt.xlabel('Yearly Return (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Inflation distribution
plt.subplot(2, 2, 4)
plt.hist((inflation_rates - 1) * 100, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribution of Yearly Inflation')
plt.xlabel('Yearly Inflation (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

nominal_tax = (final_nominal - total_contributions) * (capital_gains_tax / 100)
nominal_after_tax = final_nominal - nominal_tax

real_tax = nominal_tax / np.prod(inflation_rates)
real_after_tax = final_real - real_tax

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
cash_flows = [-c for c in contributions] 
cash_flows[-1] += final_nominal
nominal_IRR = npf.irr(cash_flows)

cash_flows[-1] = final_real
real_IRR = npf.irr(cash_flows)

cash_flows[-1] = real_after_tax
real_IRR_after_tax = npf.irr(cash_flows)

print("\nANNUALIZED RETURNS:")
print(f"Nominal IRR: {nominal_IRR * 100:.2f}%")
print(f"Real IRR: {real_IRR * 100:.2f}%")
print(f"Real IRR after tax: {real_IRR_after_tax * 100:.2f}%")
print(f"Average yearly return used: {(stats.gmean(returns)-1) * 100:.2f}%")
print(f"Average yearly inflation used: {(stats.gmean(inflation_rates)-1) * 100:.2f}%")


