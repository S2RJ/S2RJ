"""
Module 2: Probability Theory - The Language of Uncertainty
===========================================================
Learn the fundamental concepts of probability that underpin all statistical inference.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, permutations, product
from collections import Counter
import scipy.stats as stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ProbabilityBasics:
    """
    Understanding fundamental probability concepts
    """
    
    @staticmethod
    def basic_probability(favorable, total):
        """
        Basic Probability Formula: P(A) = favorable outcomes / total outcomes
        Range: 0 ≤ P(A) ≤ 1
        """
        prob = favorable / total
        print(f"P(Event) = {favorable}/{total} = {prob:.4f} = {prob*100:.2f}%")
        return prob
    
    @staticmethod
    def complement_rule(prob_a):
        """
        Complement Rule: P(not A) = 1 - P(A)
        Use: Finding probability of event NOT happening
        """
        prob_not_a = 1 - prob_a
        print(f"P(A) = {prob_a:.4f}")
        print(f"P(not A) = 1 - {prob_a:.4f} = {prob_not_a:.4f}")
        return prob_not_a
    
    @staticmethod
    def addition_rule(prob_a, prob_b, prob_both=0):
        """
        Addition Rule: P(A or B) = P(A) + P(B) - P(A and B)
        - If mutually exclusive: P(A and B) = 0
        - If independent: P(A and B) = P(A) × P(B)
        """
        prob_or = prob_a + prob_b - prob_both
        print(f"P(A or B) = P(A) + P(B) - P(A and B)")
        print(f"P(A or B) = {prob_a:.4f} + {prob_b:.4f} - {prob_both:.4f} = {prob_or:.4f}")
        return prob_or
    
    @staticmethod
    def multiplication_rule_independent(prob_a, prob_b):
        """
        Multiplication Rule (Independent Events): P(A and B) = P(A) × P(B)
        Use: When events don't affect each other
        """
        prob_and = prob_a * prob_b
        print(f"P(A and B) = P(A) × P(B)")
        print(f"P(A and B) = {prob_a:.4f} × {prob_b:.4f} = {prob_and:.4f}")
        return prob_and
    
    @staticmethod
    def conditional_probability(prob_a_and_b, prob_b):
        """
        Conditional Probability: P(A|B) = P(A and B) / P(B)
        Read as: "Probability of A given B"
        Use: When one event affects another
        """
        prob_a_given_b = prob_a_and_b / prob_b
        print(f"P(A|B) = P(A and B) / P(B)")
        print(f"P(A|B) = {prob_a_and_b:.4f} / {prob_b:.4f} = {prob_a_given_b:.4f}")
        return prob_a_given_b
    
    @staticmethod
    def bayes_theorem(prob_b_given_a, prob_a, prob_b):
        """
        Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)
        Use: Updating probabilities with new evidence
        """
        prob_a_given_b = (prob_b_given_a * prob_a) / prob_b
        print(f"Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)")
        print(f"P(A|B) = {prob_b_given_a:.4f} × {prob_a:.4f} / {prob_b:.4f}")
        print(f"P(A|B) = {prob_a_given_b:.4f}")
        return prob_a_given_b


class CombinatoricsAndCounting:
    """
    Counting methods for probability calculations
    """
    
    @staticmethod
    def factorial(n):
        """
        Factorial: n! = n × (n-1) × (n-2) × ... × 1
        Use: Counting arrangements
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        print(f"{n}! = {result:,}")
        return result
    
    @staticmethod
    def permutations_count(n, r):
        """
        Permutations: P(n,r) = n! / (n-r)!
        Use: Counting ordered arrangements
        Example: How many ways to arrange 3 books from 5?
        """
        result = np.math.factorial(n) // np.math.factorial(n - r)
        print(f"P({n},{r}) = {n}! / ({n}-{r})! = {result:,}")
        print(f"Interpretation: {result:,} ways to arrange {r} items from {n}")
        return result
    
    @staticmethod
    def combinations_count(n, r):
        """
        Combinations: C(n,r) = n! / (r! × (n-r)!)
        Use: Counting unordered selections
        Example: How many ways to choose 3 books from 5?
        """
        result = np.math.factorial(n) // (np.math.factorial(r) * np.math.factorial(n - r))
        print(f"C({n},{r}) = {n}! / ({r}! × ({n}-{r})!) = {result:,}")
        print(f"Interpretation: {result:,} ways to choose {r} items from {n}")
        return result
    
    @staticmethod
    def demonstrate_difference():
        """
        Show difference between permutations and combinations
        """
        print("\n" + "="*60)
        print("PERMUTATIONS vs COMBINATIONS")
        print("="*60)
        
        items = ['A', 'B', 'C']
        r = 2
        
        print(f"\nItems: {items}")
        print(f"Selecting {r} items\n")
        
        print("PERMUTATIONS (order matters):")
        perms = list(permutations(items, r))
        for i, p in enumerate(perms, 1):
            print(f"  {i}. {p}")
        print(f"Total: {len(perms)} arrangements")
        
        print("\nCOMBINATIONS (order doesn't matter):")
        combs = list(combinations(items, r))
        for i, c in enumerate(combs, 1):
            print(f"  {i}. {c}")
        print(f"Total: {len(combs)} selections")


class RandomVariables:
    """
    Understanding random variables and their properties
    """
    
    @staticmethod
    def expected_value(values, probabilities):
        """
        Expected Value (Mean): E(X) = Σ(x × P(x))
        Use: Average outcome in the long run
        """
        ev = np.sum(np.array(values) * np.array(probabilities))
        print(f"Expected Value E(X) = {ev:.4f}")
        print("\nCalculation:")
        for v, p in zip(values, probabilities):
            print(f"  {v} × {p:.4f} = {v*p:.4f}")
        print(f"  Sum = {ev:.4f}")
        return ev
    
    @staticmethod
    def variance_rv(values, probabilities):
        """
        Variance: Var(X) = E(X²) - [E(X)]²
        Use: Measuring spread of random variable
        """
        ev = np.sum(np.array(values) * np.array(probabilities))
        ev_squared = np.sum(np.array(values)**2 * np.array(probabilities))
        variance = ev_squared - ev**2
        std = np.sqrt(variance)
        
        print(f"E(X) = {ev:.4f}")
        print(f"E(X²) = {ev_squared:.4f}")
        print(f"Var(X) = E(X²) - [E(X)]² = {variance:.4f}")
        print(f"SD(X) = √Var(X) = {std:.4f}")
        return variance, std
    
    @staticmethod
    def visualize_probability_distribution(values, probabilities, title="Probability Distribution"):
        """
        Visualize a discrete probability distribution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PMF (Probability Mass Function)
        ax1.bar(values, probabilities, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'{title} - PMF')
        ax1.grid(True, alpha=0.3)
        
        # Add expected value line
        ev = np.sum(np.array(values) * np.array(probabilities))
        ax1.axvline(ev, color='red', linestyle='--', linewidth=2, label=f'E(X) = {ev:.2f}')
        ax1.legend()
        
        # CDF (Cumulative Distribution Function)
        cumulative = np.cumsum(probabilities)
        ax2.step(values, cumulative, where='post', color='darkgreen', linewidth=2)
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'{title} - CDF')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('/vercel/sandbox/probability_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as 'probability_distribution.png'")
        plt.show()


# ==================== PRACTICAL EXAMPLES ====================

def example_1_dice():
    """Example 1: Dice probability"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Rolling Dice")
    print("="*60)
    
    print("\n1. Probability of rolling a 6 on a fair die:")
    ProbabilityBasics.basic_probability(1, 6)
    
    print("\n2. Probability of NOT rolling a 6:")
    ProbabilityBasics.complement_rule(1/6)
    
    print("\n3. Probability of rolling a 6 OR a 5:")
    ProbabilityBasics.addition_rule(1/6, 1/6, 0)  # Mutually exclusive
    
    print("\n4. Probability of rolling two 6s in a row (independent):")
    ProbabilityBasics.multiplication_rule_independent(1/6, 1/6)
    
    # Simulate dice rolls
    print("\n5. Simulation: Rolling a die 10,000 times")
    rolls = np.random.randint(1, 7, 10000)
    counts = Counter(rolls)
    
    print("\nResults:")
    for face in range(1, 7):
        observed_prob = counts[face] / 10000
        expected_prob = 1/6
        print(f"  Face {face}: {observed_prob:.4f} (expected: {expected_prob:.4f})")


def example_2_cards():
    """Example 2: Card probability"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Drawing Cards")
    print("="*60)
    
    print("\nStandard deck: 52 cards (13 ranks × 4 suits)")
    
    print("\n1. Probability of drawing an Ace:")
    ProbabilityBasics.basic_probability(4, 52)
    
    print("\n2. Probability of drawing a Heart:")
    ProbabilityBasics.basic_probability(13, 52)
    
    print("\n3. Probability of drawing Ace OR Heart:")
    # P(Ace) + P(Heart) - P(Ace of Hearts)
    ProbabilityBasics.addition_rule(4/52, 13/52, 1/52)
    
    print("\n4. Conditional: Probability of Ace given it's a Heart:")
    # P(Ace and Heart) / P(Heart)
    ProbabilityBasics.conditional_probability(1/52, 13/52)


def example_3_medical_test():
    """Example 3: Bayes' Theorem - Medical Testing"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Medical Test (Bayes' Theorem)")
    print("="*60)
    
    print("\nScenario:")
    print("- Disease prevalence: 1% of population")
    print("- Test accuracy: 95% (detects disease when present)")
    print("- False positive rate: 5% (positive when disease absent)")
    print("\nQuestion: If you test positive, what's probability you have disease?")
    
    # Given information
    p_disease = 0.01  # P(D)
    p_no_disease = 0.99  # P(not D)
    p_pos_given_disease = 0.95  # P(+|D) - sensitivity
    p_pos_given_no_disease = 0.05  # P(+|not D) - false positive rate
    
    # Calculate P(+) using law of total probability
    p_positive = (p_pos_given_disease * p_disease + 
                  p_pos_given_no_disease * p_no_disease)
    
    print(f"\nP(Positive) = P(+|D)×P(D) + P(+|not D)×P(not D)")
    print(f"P(Positive) = {p_pos_given_disease}×{p_disease} + {p_pos_given_no_disease}×{p_no_disease}")
    print(f"P(Positive) = {p_positive:.4f}")
    
    print(f"\nNow applying Bayes' Theorem:")
    p_disease_given_pos = ProbabilityBasics.bayes_theorem(
        p_pos_given_disease, p_disease, p_positive
    )
    
    print(f"\n📊 Result: Only {p_disease_given_pos*100:.1f}% chance of having disease!")
    print("This shows why positive tests often need confirmation.")


def example_4_birthday_paradox():
    """Example 4: Birthday Paradox"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Birthday Paradox")
    print("="*60)
    
    print("\nQuestion: In a room of 23 people, what's the probability")
    print("that at least 2 people share a birthday?")
    
    print("\nSolution using complement:")
    print("P(at least 2 share) = 1 - P(all different)")
    
    n_people = 23
    n_days = 365
    
    # Calculate P(all different)
    prob_all_different = 1.0
    for i in range(n_people):
        prob_all_different *= (n_days - i) / n_days
    
    prob_at_least_two = 1 - prob_all_different
    
    print(f"\nP(all different) = {prob_all_different:.4f}")
    print(f"P(at least 2 share) = {prob_at_least_two:.4f} = {prob_at_least_two*100:.1f}%")
    
    # Simulate
    print("\nSimulation with 10,000 rooms of 23 people:")
    matches = 0
    for _ in range(10000):
        birthdays = np.random.randint(0, 365, n_people)
        if len(birthdays) != len(set(birthdays)):
            matches += 1
    
    print(f"Simulated probability: {matches/10000:.4f} = {matches/100:.1f}%")
    
    # Show how probability changes with group size
    sizes = range(2, 51)
    probs = []
    for n in sizes:
        prob_diff = 1.0
        for i in range(n):
            prob_diff *= (365 - i) / 365
        probs.append(1 - prob_diff)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, probs, linewidth=2, color='darkblue')
    plt.axhline(0.5, color='red', linestyle='--', label='50% probability')
    plt.axvline(23, color='green', linestyle='--', label='23 people')
    plt.xlabel('Number of People')
    plt.ylabel('Probability of Shared Birthday')
    plt.title('Birthday Paradox: Probability vs Group Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('/vercel/sandbox/birthday_paradox.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'birthday_paradox.png'")
    plt.show()


def example_5_expected_value():
    """Example 5: Expected Value in Games"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Expected Value - Casino Game")
    print("="*60)
    
    print("\nGame: Roll a die")
    print("- Roll 6: Win $10")
    print("- Roll 4 or 5: Win $5")
    print("- Roll 1, 2, or 3: Lose $3")
    print("- Cost to play: $2")
    
    values = [10, 5, 5, -3, -3, -3]
    probabilities = [1/6] * 6
    
    print("\nExpected winnings per roll:")
    ev_winnings = RandomVariables.expected_value(values, probabilities)
    
    print(f"\nExpected profit (after $2 cost): ${ev_winnings - 2:.2f}")
    
    if ev_winnings - 2 > 0:
        print("✓ Positive expected value - good game to play!")
    else:
        print("✗ Negative expected value - you'll lose money over time")
    
    # Visualize
    RandomVariables.visualize_probability_distribution(
        values, probabilities, "Casino Game Outcomes"
    )


def example_6_counting():
    """Example 6: Counting and Combinatorics"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Counting Problems")
    print("="*60)
    
    print("\n1. How many ways to arrange 5 books on a shelf?")
    CombinatoricsAndCounting.factorial(5)
    
    print("\n2. How many ways to select 3 people from 10 for a committee?")
    CombinatoricsAndCounting.combinations_count(10, 3)
    
    print("\n3. How many ways to award gold, silver, bronze to 8 runners?")
    CombinatoricsAndCounting.permutations_count(8, 3)
    
    print("\n4. Lottery: Choose 6 numbers from 49")
    total_combinations = CombinatoricsAndCounting.combinations_count(49, 6)
    prob_win = 1 / total_combinations
    print(f"Probability of winning: {prob_win:.10f} = 1 in {total_combinations:,}")
    
    # Demonstrate difference
    CombinatoricsAndCounting.demonstrate_difference()


# ==================== INTERACTIVE EXERCISES ====================

def exercise_1():
    """Exercise 1: Basic probability"""
    print("\n" + "="*60)
    print("EXERCISE 1: Calculate Probabilities")
    print("="*60)
    
    print("\nScenario: A bag contains 5 red, 3 blue, and 2 green marbles")
    print("\nQuestions:")
    print("1. P(red)?")
    print("2. P(not red)?")
    print("3. P(red or blue)?")
    print("4. P(red, then blue without replacement)?")
    
    print("\n--- SOLUTIONS ---")
    total = 10
    
    print("\n1. P(red):")
    p_red = ProbabilityBasics.basic_probability(5, total)
    
    print("\n2. P(not red):")
    p_not_red = ProbabilityBasics.complement_rule(p_red)
    
    print("\n3. P(red or blue):")
    p_red_or_blue = ProbabilityBasics.addition_rule(5/10, 3/10, 0)
    
    print("\n4. P(red then blue):")
    print("First draw red: 5/10")
    print("Then draw blue: 3/9 (one marble removed)")
    p_red_then_blue = (5/10) * (3/9)
    print(f"P(red then blue) = 5/10 × 3/9 = {p_red_then_blue:.4f}")


def exercise_2():
    """Exercise 2: Expected value"""
    print("\n" + "="*60)
    print("EXERCISE 2: Expected Value")
    print("="*60)
    
    print("\nScenario: Investment opportunity")
    print("- 30% chance: Gain $1000")
    print("- 50% chance: Gain $500")
    print("- 20% chance: Lose $200")
    print("\nQuestion: What's the expected return?")
    
    print("\n--- SOLUTION ---")
    values = [1000, 500, -200]
    probabilities = [0.30, 0.50, 0.20]
    
    ev = RandomVariables.expected_value(values, probabilities)
    print(f"\nExpected return: ${ev:.2f}")
    
    if ev > 0:
        print("✓ Positive expected value - potentially good investment")


def exercise_3():
    """Exercise 3: Conditional probability"""
    print("\n" + "="*60)
    print("EXERCISE 3: Conditional Probability")
    print("="*60)
    
    print("\nScenario: Weather forecast")
    print("- P(Rain) = 0.3")
    print("- P(Traffic | Rain) = 0.8")
    print("- P(Traffic | No Rain) = 0.2")
    print("\nQuestions:")
    print("1. P(Traffic)?")
    print("2. P(Rain | Traffic)?")
    
    print("\n--- SOLUTIONS ---")
    p_rain = 0.3
    p_no_rain = 0.7
    p_traffic_given_rain = 0.8
    p_traffic_given_no_rain = 0.2
    
    print("\n1. P(Traffic) using law of total probability:")
    p_traffic = p_traffic_given_rain * p_rain + p_traffic_given_no_rain * p_no_rain
    print(f"P(Traffic) = {p_traffic_given_rain}×{p_rain} + {p_traffic_given_no_rain}×{p_no_rain}")
    print(f"P(Traffic) = {p_traffic:.4f}")
    
    print("\n2. P(Rain | Traffic) using Bayes' Theorem:")
    p_rain_given_traffic = ProbabilityBasics.bayes_theorem(
        p_traffic_given_rain, p_rain, p_traffic
    )


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "🎲 " * 20)
    print("WELCOME TO MODULE 2: PROBABILITY THEORY")
    print("🎲 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Understand basic probability rules and axioms")
    print("2. Learn conditional probability and independence")
    print("3. Master Bayes' Theorem")
    print("4. Apply counting methods (permutations & combinations)")
    print("5. Calculate expected values and variance")
    print("6. Solve real-world probability problems")
    
    # Run examples
    example_1_dice()
    example_2_cards()
    example_3_medical_test()
    example_4_birthday_paradox()
    example_5_expected_value()
    example_6_counting()
    
    # Run exercises
    exercise_1()
    exercise_2()
    exercise_3()
    
    print("\n" + "="*60)
    print("✅ MODULE 2 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Probability quantifies uncertainty (0 to 1)")
    print("• Use complement rule for 'not' events")
    print("• Independent events: P(A and B) = P(A) × P(B)")
    print("• Bayes' Theorem updates probabilities with new evidence")
    print("• Expected value = long-run average outcome")
    print("• Permutations: order matters, Combinations: order doesn't")
    print("\n➡️  Next: Module 3 - Probability Distributions")
