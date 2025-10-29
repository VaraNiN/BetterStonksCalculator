import numpy as np
import numpy_financial as npf
import scipy.stats as stats
import matplotlib.pyplot as plt

##### This is amateur code. Use at your own risk. #####
##### Do not use for real investment decisions. #####
##### No financial advice is given or implied. #####

years = 30 # Investment period
starting_sum = 0 # In €
monthly_contribution = 500  # In €
average_return = 7 # In percent
average_inflation = 2 # In percent

# To simulate salary (and thus contribution) increasing with inflation
increase_contribution_with_inflation = False

# Instead of using the usual constant average return and inflation rates,
# do you want to use a more realistic distribution-based approach? (Recommended)
use_distribution_based_returns = True

minimum_return = -50 # In percent
maximum_return = 40 # In percent
freedom_return = 0.1    # How far the returns can deviate from the average (lower = more variance)

minimum_inflation = -1 # In percent
maximum_inflation = 15 # In percent
freedom_inflation = 0.5 # How far the returns can deviate from the average (lower = more variance)

force_exact_return = True   # (Recommended) Forces the geometric mean of the sampled returns to exactly equal average_return
force_exact_inflation = True # (Recommended) Forces the geometric mean of the sampled inflation to exactly equal average_inflation
fixed_seed = 0          # If non-zero, uses this seed for random number generation (for reproducibility)

def student_t(x, df=2, loc=2, minimum=0, maximum=10):
    """
    Student's t-distribution PDF, limited and normalized over [minimum, maximum]
    df: degrees of freedom
    loc: center
    scale: scale parameter
    """
    result = stats.t.pdf(x, df, loc=loc)
    result = np.where((x >= minimum) & (x <= maximum), result, 0)
    x_norm = np.linspace(minimum, maximum, 10000)
    norm_vals = stats.t.pdf(x_norm, df, loc=loc)
    area = np.trapz(norm_vals, x_norm)
    return result / area

def sample_student_t(n_samples, df=2, loc=2, minimum=0, maximum=10, force_exact=False, fixed_seed=True):
    """
    Sample from truncated Student's t-distribution using rejection sampling
    """
    samples = []
    if fixed_seed:
        np.random.seed(42)
    x_test = np.linspace(minimum, maximum, n_samples*10)
    y_test = student_t(x_test, df, loc, minimum, maximum)
    max_value = np.max(y_test)
    while len(samples) < n_samples:
        x_candidate = np.random.uniform(minimum, maximum)
        y_candidate = student_t(np.array([x_candidate]), df, loc, minimum, maximum)[0]
        if np.random.uniform(0, max_value) < y_candidate:
            samples.append(x_candidate)
    results = 1 + np.array(samples)/100.
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

def plot_results(portfolio_value, real_value, yearly_returns, yearly_inflation, total_contributions):
    """Plot the simulation results (yearly steps)"""
    years_axis = np.arange(len(portfolio_value))
    
    plt.figure(figsize=(15, 12))
    
    # Portfolio value over time
    plt.subplot(2, 2, 1)
    plt.plot(years_axis, portfolio_value, 'b-', linewidth=2, label='Nominal Value')
    plt.plot(years_axis, real_value, 'r-', linewidth=2, label='Real Value (Inflation-Adjusted)')
    plt.plot(years_axis, [total_contributions * (i / years) for i in years_axis], 'k--', linewidth=1, label='Total Contributions')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Years')
    plt.ylabel('Portfolio Value (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final summary
    plt.subplot(2, 2, 2)
    final_nominal = portfolio_value[-1]
    final_real = real_value[-1]
    
    categories = ['Total\nContributions', 'Final Nominal\nValue', 'Final Real\nValue']
    values = [total_contributions, final_nominal, final_real]
    colors = ['gray', 'blue', 'red']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    # Add value labels on bars
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
    
    return final_nominal, final_real, total_contributions

# Run the investment simulation
print("Running Better Stonks Calculator...")
print(f"Initial investment: €{starting_sum:,}")
print(f"Yearly contribution: €{monthly_contribution * 12:,}")
print(f"Investment period: {years} years")
print(f"Expected return: {average_return}% (range: {minimum_return}% to {maximum_return}%)")
print(f"Expected inflation: {average_inflation}% (range: {minimum_inflation}% to {maximum_inflation}%)")
print(f"Force exact averages: Returns={force_exact_return}, Inflation={force_exact_inflation}")
print("-" * 60)

# Run simulation
portfolio_values, real_values, returns, inflation_rates, total_contributions, contributions = run_investment_simulation()

# Plot results
final_nominal, final_real, total_contributions = plot_results(portfolio_values, real_values, returns, inflation_rates, total_contributions)

# Print final statistics
print(f"\nFINAL RESULTS AFTER {years} YEARS:")
print(f"Total contributions: €{total_contributions:,.0f}")
print(f"Final nominal value: €{final_nominal:,.0f}")
print(f"Final real value (inflation-adjusted): €{final_real:,.0f}")
print(f"Nominal gain: €{final_nominal - total_contributions:,.0f}")
print(f"Real gain: €{final_real - total_contributions:,.0f}")
print(f"Nominal return multiple: {final_nominal / total_contributions:.2f}x")
print(f"Real return multiple: {final_real / total_contributions:.2f}x")

# Build cash flow arrays for IRR calculation
nominal_cash_flows = [-c for c in contributions] 
nominal_cash_flows[-1] += final_nominal
nominal_IRR = npf.irr(nominal_cash_flows)

real_cash_flows = [-c for c in contributions]
real_cash_flows[-1] += final_real
real_IRR = npf.irr(real_cash_flows)

print("\nANNUALIZED RETURNS:")
print(f"Nominal IRR: {nominal_IRR * 100:.2f}%")
print(f"Real IRR: {real_IRR * 100:.2f}%")
print(f"Average yearly return used: {(stats.gmean(returns)-1) * 100:.2f}%")
print(f"Average yearly inflation used: {(stats.gmean(inflation_rates)-1) * 100:.2f}%")


