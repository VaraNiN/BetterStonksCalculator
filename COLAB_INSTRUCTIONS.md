# How to Create a Google Colab Notebook

## Step-by-Step Guide

### 1. Create the Notebook File

Create a new file named `BetterStonksCalculator.ipynb` with the following structure:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Better Stonks Calculator 📈\n",
    "\n",
    "**⚠️ WARNING**: This is for educational purposes only. Not financial advice!\n",
    "\n",
    "This calculator simulates investment scenarios using real historical data.\n",
    "\n",
    "## How to Use:\n",
    "1. Click **Runtime → Run all** to run everything\n",
    "2. Adjust parameters in the cell below\n",
    "3. Run cells individually with Shift+Enter\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages (only needed in Colab)\n",
    "!pip install numpy-financial yfinance -q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download inflation data\n",
    "!mkdir -p data\n",
    "!wget -q -O data/inflation.csv https://raw.githubusercontent.com/VaraNiN/BetterStonksCalculator/main/data/inflation.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎮 Adjust Your Parameters Here\n",
    "\n",
    "Change the values below to match your scenario:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# YOUR PARAMETERS - CHANGE THESE!\n",
    "ticker_symbol = 'SPY'              # Stock/ETF to simulate\n",
    "years = 25                         # How many years?\n",
    "starting_sum = 0                   # Initial investment (€)\n",
    "monthly_contribution = 500         # Monthly contribution (€)\n",
    "capital_gains_tax = 27.5          # Tax rate (%)\n",
    "simulation_iteration = 100         # Number of simulations (100=fast, 1000=smooth)\n",
    "increase_contribution_with_inflation = True  # Salary increases with inflation?\n",
    "\n",
    "use_constant_returns = False       # Use fixed returns?\n",
    "average_return = 13.06            # Expected annual return (%)\n",
    "average_inflation = 2.56          # Expected annual inflation (%)\n",
    "scramble_for_sample = False       # Randomize history vs use recent years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Copy the entire run.py content here (everything from imports to the end)\n",
    "# REPLACE THIS COMMENT WITH THE ACTUAL CODE FROM run.py"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

### 2. Alternative: Use This Simple Template

Or create a simpler version by just pasting your code:

**Cell 1** (Markdown):
```markdown
# Better Stonks Calculator
Instructions: Run all cells, then adjust parameters and re-run.
```

**Cell 2** (Code):
```python
!pip install numpy-financial yfinance -q
!mkdir -p data
!wget -q -O data/inflation.csv https://raw.githubusercontent.com/VaraNiN/BetterStonksCalculator/main/data/inflation.csv
```

**Cell 3** (Code):
```python
# Paste your entire run.py code here
```

### 3. Upload to GitHub

1. Save the `.ipynb` file
2. Commit and push to your repository:
   ```bash
   git add BetterStonksCalculator.ipynb
   git add README.md
   git commit -m "Add Colab notebook and README"
   git push
   ```

### 4. Share the Colab Link

Your Colab link will be:
```
https://colab.research.google.com/github/VaraNiN/BetterStonksCalculator/blob/main/BetterStonksCalculator.ipynb
```

## Tips for Colab Users

- **Runtime → Run all**: Runs everything at once
- **Runtime → Restart runtime**: Start fresh if something breaks
- **File → Save a copy in Drive**: Save their modified version
- Colab runs in the cloud - no installation needed!
- Each user gets their own isolated copy

## What to Tell Users

"Just click this link and press 'Runtime → Run all'. Wait a few seconds for it to complete, then you can see the results. Want to try different scenarios? Change the numbers in the second cell and run again!"
