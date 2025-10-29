import numpy as np
import matplotlib.pyplot as plt

starting_sum = 5000
monthly_contribution = 500
years = 30

minimum_return = -10
average_return = 7
maximum_return = 20
return_gamma = 0.1

minimum_inflation = 0
average_inflation = 2
maximum_inflation = 10
inflation_gamma = 0.5

force_exact_return = True
force_exact_inflation = True
fixed_seed = False

def cauchy(x, x0=2, gamma=1, minimum=0, maximum=10):
    """
    Cauchy distribution (Lorentzian) centered at x0 with scale parameter gamma
    Limited to range [minimum, maximum]
    """
    # Calculate Cauchy distribution
    result = (gamma / np.pi) / ((x - x0)**2 + gamma**2)

    # Limit to range [minimum, maximum]
    result = np.where((x >= minimum) & (x <= maximum), result, 0)

    # Normalize so area integrates to 1 over [minimum, maximum]
    x_norm = np.linspace(minimum, maximum, 10000)
    norm_vals = (gamma / np.pi) / ((x_norm - x0)**2 + gamma**2)
    area = np.trapz(norm_vals, x_norm)
    
    return result / area

def sample_cauchy(n_samples, x0=2, gamma=1, minimum=0, maximum=10, force_exact=False, fixed_seed=True):
    """
    Sample from truncated Cauchy distribution using rejection sampling
    """
    samples = []
    if fixed_seed:
        np.random.seed(42)  # For reproducible results
    
    # Find maximum value for rejection sampling
    x_test = np.linspace(minimum, maximum, 1000)
    y_test = cauchy(x_test, x0, gamma, minimum, maximum)
    max_value = np.max(y_test)

    while len(samples) < n_samples:
        # Generate random x in [minimum, maximum]
        x_candidate = np.random.uniform(minimum, maximum)
        
        # Calculate function value
        y_candidate = cauchy(np.array([x_candidate]), x0, gamma, minimum, maximum)[0]
        
        # Accept with probability proportional to function value
        if np.random.uniform(0, max_value) < y_candidate:
            samples.append(x_candidate)

    results = np.array(samples)
    
    # Force exact mean if requested
    if force_exact:
        results = results * (x0 / np.mean(results))

    return results

def generate_returns(years, months_per_year=12):
    """Generate monthly returns using Cauchy distribution"""
    n_months = years * months_per_year
    return sample_cauchy(
        n_samples=n_months,
        x0=average_return,
        gamma=return_gamma,
        minimum=minimum_return,
        maximum=maximum_return,
        force_exact=force_exact_return,
        fixed_seed=fixed_seed
    )

def generate_inflation(years, months_per_year=12):
    """Generate monthly inflation rates using Cauchy distribution"""
    n_months = years * months_per_year
    return sample_cauchy(
        n_samples=n_months,
        x0=average_inflation,
        gamma=inflation_gamma,
        minimum=minimum_inflation,
        maximum=maximum_inflation,
        force_exact=force_exact_inflation,
        fixed_seed=fixed_seed
    )

def run_investment_simulation():
    """Run the complete investment simulation"""
    n_months = years * 12
    
    # Generate random returns and inflation for each month
    monthly_returns = generate_returns(years) / 100  # Convert percentage to decimal
    monthly_inflation = generate_inflation(years) / 100  # Convert percentage to decimal
    
    # Initialize tracking arrays
    portfolio_value = np.zeros(n_months + 1)
    real_value = np.zeros(n_months + 1)  # Inflation-adjusted value
    portfolio_value[0] = starting_sum
    real_value[0] = starting_sum
    
    # Cumulative inflation factor
    inflation_factor = 1.0
    
    # Run simulation month by month
    for month in range(n_months):
        # Apply monthly return to current portfolio
        portfolio_value[month + 1] = portfolio_value[month] * (1 + monthly_returns[month] / 12)
        
        # Add monthly contribution
        portfolio_value[month + 1] += monthly_contribution
        
        # Update inflation factor
        inflation_factor *= (1 + monthly_inflation[month] / 12)
        
        # Calculate real (inflation-adjusted) value
        real_value[month + 1] = portfolio_value[month + 1] / inflation_factor
    
    return portfolio_value, real_value, monthly_returns, monthly_inflation

def plot_results(portfolio_value, real_value, monthly_returns, monthly_inflation):
    """Plot the simulation results"""
    months = np.arange(len(portfolio_value))
    years_axis = months / 12
    
    plt.figure(figsize=(15, 12))
    
    # Portfolio value over time
    plt.subplot(2, 2, 1)
    plt.plot(years_axis, portfolio_value, 'b-', linewidth=2, label='Nominal Value')
    plt.plot(years_axis, real_value, 'r-', linewidth=2, label='Real Value (Inflation-Adjusted)')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Years')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 2)
    plt.hist(monthly_returns * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution of Monthly Returns')
    plt.xlabel('Monthly Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Inflation distribution
    plt.subplot(2, 2, 3)
    plt.hist(monthly_inflation * 100, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of Monthly Inflation')
    plt.xlabel('Monthly Inflation (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Final summary
    plt.subplot(2, 2, 4)
    final_nominal = portfolio_value[-1]
    final_real = real_value[-1]
    total_contributions = starting_sum + (monthly_contribution * years * 12)
    
    categories = ['Total\nContributions', 'Final Nominal\nValue', 'Final Real\nValue']
    values = [total_contributions, final_nominal, final_real]
    colors = ['gray', 'blue', 'red']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Final Results Summary')
    plt.ylabel('Value ($)')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${value:,.0f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return final_nominal, final_real, total_contributions

# Run the investment simulation
print("Running Better Stonks Calculator...")
print(f"Initial investment: ${starting_sum:,}")
print(f"Monthly contribution: ${monthly_contribution:,}")
print(f"Investment period: {years} years")
print(f"Expected return: {average_return}% (range: {minimum_return}% to {maximum_return}%)")
print(f"Expected inflation: {average_inflation}% (range: {minimum_inflation}% to {maximum_inflation}%)")
print(f"Force exact averages: Returns={force_exact_return}, Inflation={force_exact_inflation}")
print("-" * 60)

# Run simulation
portfolio_values, real_values, returns, inflation_rates = run_investment_simulation()

# Plot results
final_nominal, final_real, total_contributions = plot_results(portfolio_values, real_values, returns, inflation_rates)

# Print final statistics
print(f"\nFINAL RESULTS AFTER {years} YEARS:")
print(f"Total contributions: ${total_contributions:,.0f}")
print(f"Final nominal value: ${final_nominal:,.0f}")
print(f"Final real value (inflation-adjusted): ${final_real:,.0f}")
print(f"Nominal gain: ${final_nominal - total_contributions:,.0f}")
print(f"Real gain: ${final_real - total_contributions:,.0f}")
print(f"Nominal return multiple: {final_nominal / total_contributions:.2f}x")
print(f"Real return multiple: {final_real / total_contributions:.2f}x")

# Calculate annualized returns
nominal_annual_return = (final_nominal / starting_sum) ** (1/years) - 1
real_annual_return = (final_real / starting_sum) ** (1/years) - 1

print("\nANNUALIZED RETURNS:")
print(f"Nominal annual return: {nominal_annual_return * 100:.2f}%")
print(f"Real annual return: {real_annual_return * 100:.2f}%")
print(f"Average monthly return used: {np.mean(returns) * 100:.2f}%")
print(f"Average monthly inflation used: {np.mean(inflation_rates) * 100:.2f}%")


