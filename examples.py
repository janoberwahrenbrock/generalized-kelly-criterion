"""
Examples of using the Generalized Kelly Criterion implementation.

This script demonstrates:
1. A manipulated coin toss game (discrete probabilities)
2. A Gaussian probability distribution with a continuous return function.
"""

from generalizedKellyCriterion import GeneralizedKellyCriterion, OutcomeSet, OutcomeInterval
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from scipy.stats import norm
from scipy.integrate import quad



def plot_growth_factors_for_f_values(growth_factors_for_f_values: Dict[float, float]):
    """
    Plots the growth factors as a function of bet fractions (f values).

    Args:
        growth_factors_for_f_values (Dict[float, float]): Dictionary where keys are bet fractions (f values)
                                                          and values are corresponding growth factors.
    """
    # Sort the dictionary by f values to ensure the plot is ordered
    sorted_growth_factors = dict(sorted(growth_factors_for_f_values.items()))
    f_list = list(sorted_growth_factors.keys())
    g_list = list(sorted_growth_factors.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(f_list, g_list, label="Growth Factor", marker='o', markersize=3)
    
    # Add labels, title, and legend
    plt.xlabel("Bet Fraction (f)", fontsize=12)
    plt.ylabel("Growth Factor", fontsize=12)
    plt.title("Growth Factors for Bet Fractions", fontsize=14)
    plt.axhline(y=1, color='red', linestyle='--', label="Break-even Growth Factor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()



def manipulated_coin_toss_example():
    """
    Demonstrates GeneralizedKellyCriterion usage with a manipulated coin tossing game.
    """
    print("\n--- Running Manipulated Coin Toss Example ---")

    # Define the outcomes for the coin toss
    outcomes = ["Heads", "Tails"]
    
    # Define the probability mass function (manipulated probabilities)
    p = lambda x: {"Heads": 0.6, "Tails": 0.4}.get(x, 0)  # 60% for Heads, 40% for Tails
    
    # Define the return function for each outcome
    r = lambda x: {"Heads": 1, "Tails": -1}.get(x, 0)  # 1x return on Heads, -1x on Tails
    
    # Create a discrete domain (OutcomeSet) with the outcomes
    domain = OutcomeSet(outcomes)
    
    # Create a GeneralizedKellyCriterion instance
    kelly = GeneralizedKellyCriterion(r, p, domain)
    
    # Calculate the optimal bet fraction
    optimal_result = kelly.calculate_optimal_f()
    print(f"Optimal bet fraction (f*): {optimal_result['optimal_f']:.4f}")
    print(f"Expected growth factor: {optimal_result['growth_factor']:.4f}")

    # get min_f and max_f
    min_f = -1/kelly.max_value_of_r * 0.999
    max_f = -1/kelly.min_value_of_r * 0.999

    # Calculate growth factors for specific bet fractions
    f_values = np.linspace(min_f, max_f, 1000)
    growth_factors_for_f_values = kelly.calculate_growth_factors_for_f_values(f_values)
    
    # Plot the results
    plot_growth_factors_for_f_values(growth_factors_for_f_values)



def gaussian_experiment():
    """
    Demonstrates GeneralizedKellyCriterion usage with a Gaussian probability density function and a cubic return function.
    """
    print("\n--- Running Gaussian Distribution Experiment ---")

    # Define the Gaussian distribution parameters
    mean = 0.1
    std_dev = 1

    # Calculate normalization factor
    Z, _ = quad(lambda x: norm.pdf(x, loc=mean, scale=std_dev), -3, 3)

    # Define the probability density function (PDF) of the Gaussian distribution
    p = lambda x: norm.pdf(x, loc=mean, scale=std_dev) / Z if -3 <= x <= 3 else 0

    # Define the return function
    def r(x):
        if x < -1:
            return -1  # Total loss
        elif x > 1:
            return 1   # Maximal gain
        else:
            return x   # Linear return within [-1,1]

    # Define the continuous domain as an OutcomeInterval from -5 to 5
    domain = OutcomeInterval(-3, 3)

    # Create a GeneralizedKellyCriterion instance
    kelly = GeneralizedKellyCriterion(r, p, domain)

    # Get the optimal bet fraction
    optimal_result = kelly.calculate_optimal_f()
    print(f"Optimal bet fraction (f*): {optimal_result['optimal_f']:.4f}")
    print(f"Expected growth factor: {optimal_result['growth_factor']:.4f}")

    # get min_f and max_f
    min_f = -1 / kelly.max_value_of_r * 0.999
    max_f = -1 / kelly.min_value_of_r * 0.999

    # Calculate growth factors for specific bet fractions
    f_values = np.linspace(min_f, max_f, 1000)
    growth_factors_for_f_values = kelly.calculate_growth_factors_for_f_values(f_values)

    # Plot the results
    plot_growth_factors_for_f_values(growth_factors_for_f_values)



if __name__ == "__main__":
    
    # Run both examples
    manipulated_coin_toss_example()
    gaussian_experiment()