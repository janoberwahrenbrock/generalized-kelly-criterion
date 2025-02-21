import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from typing import List, Dict, Callable, Union


class OutcomeInterval:
    """
    Represents a closed interval [lowerbound, upperbound].
    Used to define continuous domains for probability density functions and return functions.
    """

    def __init__(self, lowerbound, upperbound):
        if lowerbound > upperbound:
            raise ValueError("Lowerbound must be less than or equal to upperbound.")
        
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def __repr__(self):
        return f"OutcomeInterval(lowerbound={self.lowerbound}, upperbound={self.upperbound})"


class OutcomeSet:
    """
    Represents a finite set of discrete outcomes.
    Used to define discrete domains for probability mass functions and return functions.
    """

    def __init__(self, items=None):
        self.set = set(items) if items else set()

    def __repr__(self):
        return f"OutcomeSet(set={sorted(self.set)})"
    

class GeneralizedKellyCriterion: 
    """
    Implements the generalized Kelly criterion for optimal bet sizing.
    Can be used with continuous (OutcomeInterval) or discrete (OutcomeSet) domains.
    """

    def __init__(self, r: Callable[[float], float], p: Callable[[float], float], domain: Union[OutcomeInterval, OutcomeSet]):
        self.r = r
        self.p = p
        self.domain = domain

        # Check that r and p are callable functions
        if not callable(self.p) or not callable(self.r):
            raise TypeError("self.p and self.r must be callable functions.")

        # Check that the domain is an OutcomeInterval or a OutcomeSet
        if not isinstance(domain, OutcomeInterval) and not isinstance(domain, OutcomeSet):
            raise TypeError("The domain must be an OutcomeInterval or a OutcomeSet.")

        if isinstance(domain, OutcomeInterval):
            # Check that the function p has no negative values within the interval
            min_value_of_p = minimize_scalar(p, bounds=(domain.lowerbound, domain.upperbound), method='bounded').fun
            if min_value_of_p < 0:
                raise ValueError(f"The function p has negative values. Minimum value found: {min_value_of_p}.")

            # Check that the probability sum equals 1
            result, abserr = quad(p, domain.lowerbound, domain.upperbound)
            if not np.isclose(result, 1, atol=1e-6):
                raise ValueError(f"The probability sum is {result}, but it must be close to 1.")

            # calculate min(r(x))
            min_value_of_r = minimize_scalar(r, bounds=(domain.lowerbound, domain.upperbound), method='bounded').fun
            self.min_value_of_r = min_value_of_r

            # calculate max(r(x))
            max_value_of_r = -minimize_scalar(lambda x: -r(x), bounds=(domain.lowerbound, domain.upperbound), method='bounded').fun
            self.max_value_of_r = max_value_of_r

        elif isinstance(domain, OutcomeSet):
            prob_list = np.array([p(x) for x in domain.set]) # List of all occuring probabilities

            # Check that the function p has no negative values within the OutcomeSet
            if np.any(prob_list < 0):
                negative_values = prob_list[prob_list < 0]
                raise ValueError(f"The function p has negative values within the OutcomeSet. Negative values found: {negative_values}.")

            # Check that the probability sum equals 1
            prob_sum = np.sum(prob_list)
            if not np.isclose(prob_sum, 1, atol=1e-6):
                raise ValueError(f"The probability sum is {prob_sum}, but it must be close to 1.")

            r_values = np.array([r(x) for x in domain.set])

            # Calculate min(r(x)) for the set
            min_value_of_r = r_values.min()
            self.min_value_of_r = min_value_of_r

            # Calculate max(r(x)) for the set
            max_value_of_r = r_values.max()
            self.max_value_of_r = max_value_of_r


    def calculate_optimal_f(self, min_f=-np.inf, max_f=np.inf) -> Dict[str, float]:
        """
        Calculates the optimal fraction to bet using the Kelly criterion.

        Returns:
            Dict[str, float]: 
                - "optimal_f" (float): The fraction of capital to bet for maximum growth.
                - "growth_factor" (float): The expected growth factor with this bet size.
        """

        if self.max_value_of_r == 0 or self.min_value_of_r == 0:
            raise ValueError("max(r) and min(r) must not be zero to avoid division by zero.")

        # Check that min_f and max_f are in the interval [-1/r_max, -1/r_min]
        if min_f < -1/self.max_value_of_r:
            min_f = -1/self.max_value_of_r * 0.999
            print(f"min_f was set to {min_f} due to the maximum value of r being {self.max_value_of_r}.")
        if max_f > -1/self.min_value_of_r:
            max_f = -1/self.min_value_of_r * 0.999
            print(f"max_f was set to {max_f} due to the minimum value of r being {self.min_value_of_r}.")

        # Define the function to be optimized based on the domain type
        if isinstance(self.domain, OutcomeInterval):
            def func(f):
                integrand = lambda x: self.p(x) * np.log(1 + f * self.r(x))
                result, abserr = quad(integrand, self.domain.lowerbound, self.domain.upperbound)
                return result
        elif isinstance(self.domain, OutcomeSet):
            def func(f):
                return np.sum([self.p(x) * np.log(1 + f * self.r(x)) for x in self.domain.set])

        # Maximize the function
        neg_func = lambda f: -func(f) # instead of maximizing the function, we minimize the negative function
        result = minimize_scalar(neg_func, bounds=(min_f, max_f), method="bounded") # returns an OptimizeResult object

        if result.success:
            optimal_f = result.x # The solution of the optimization
            min_value_of_neg_func = result.fun # Value of objective function at x
        else:
            raise ValueError("Optimization was not successful.")
        
        growth_factor = np.exp(-min_value_of_neg_func) # The growth factor is the exponential of the negative function value
        return {"optimal_f": optimal_f, "growth_factor": growth_factor}


    def calculate_growth_factors_for_f_values(self, list_of_f_values: List[float]) -> Dict[float, float]:
        """
        Calculates the growth factors for a given list of bet fractions.

        Args:
            list_of_f_values (List[float]): A list of bet fractions (values between 0 and 1).

        Returns:
            Dict[float, float]: A dictionary where each key is a bet fraction, and 
                                the value is the corresponding growth factor.
        """
        # Adjust min_f and max_f based on r values
        limit_min_f = -1 / self.max_value_of_r
        limit_max_f = -1 / self.min_value_of_r

        # Check that the f values are in the interval [min_f, max_f]
        if not np.all((limit_min_f < np.array(list_of_f_values)) & (np.array(list_of_f_values) < limit_max_f)):
            raise ValueError(f"The f values must be in the open interval ({limit_min_f}, {limit_max_f}) due to the minimum and maximum values of r being {self.min_value_of_r} and {self.max_value_of_r}.")

        growth_factors = {}

        # Define the function based on the domain type
        if isinstance(self.domain, OutcomeInterval):
            def func(f):
                integrand = lambda x: self.p(x) * np.log(1 + f * self.r(x))
                integration_result, abserr = quad(integrand, self.domain.lowerbound, self.domain.upperbound)
                return np.exp(integration_result)
        elif isinstance(self.domain, OutcomeSet):
            def func(f):
                return np.exp(np.sum([self.p(x) * np.log(1 + f * self.r(x)) for x in self.domain.set]))

        # alculating growth factors for each f value
        for f in tqdm(list_of_f_values, desc="Calculating growth factors"):
            try:
                growth_factors[f] = func(f)
            except Exception as e:
                raise ValueError(f"Error occurred while calculating growth factor for f={f}.") from e

        return growth_factors