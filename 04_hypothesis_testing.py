"""
Module 4: Hypothesis Testing - Making Data-Driven Decisions
============================================================
Learn how to test claims and make statistical inferences from data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HypothesisTestingBasics:
    """
    Understanding the fundamentals of hypothesis testing
    """
    
    @staticmethod
    def explain_concepts():
        """Explain core concepts"""
        print("\n" + "="*60)
        print("HYPOTHESIS TESTING FUNDAMENTALS")
        print("="*60)
        
        print("\n1. NULL HYPOTHESIS (H₀):")
        print("   • The status quo or default assumption")
        print("   • What we assume is true until proven otherwise")
        print("   • Example: 'The drug has no effect'")
        
        print("\n2. ALTERNATIVE HYPOTHESIS (H₁ or Hₐ):")
        print("   • What we're trying to prove")
        print("   • The research hypothesis")
        print("   • Example: 'The drug has an effect'")
        
        print("\n3. SIGNIFICANCE LEVEL (α):")
        print("   • Probability of rejecting H₀ when it's true (Type I error)")
        print("   • Common values: 0.05, 0.01, 0.10")
        print("   • α = 0.05 means 5% chance of false positive")
        
        print("\n4. P-VALUE:")
        print("   • Probability of observing data this extreme if H₀ is true")
        print("   • If p-value < α: Reject H₀ (statistically significant)")
        print("   • If p-value ≥ α: Fail to reject H₀ (not significant)")
        
        print("\n5. TEST STATISTIC:")
        print("   • Standardized value calculated from sample data")
        print("   • Examples: z-score, t-statistic, chi-square")
        
        print("\n6. TYPES OF ERRORS:")
        print("   • Type I Error (α): Reject true H₀ (false positive)")
        print("   • Type II Error (β): Fail to reject false H₀ (false negative)")
        print("   • Power (1-β): Probability of correctly rejecting false H₀")
        
        print("\n7. ONE-TAILED vs TWO-TAILED:")
        print("   • One-tailed: Testing if parameter is greater OR less than value")
        print("   • Two-tailed: Testing if parameter is different from value")
    
    @staticmethod
    def visualize_hypothesis_test(test_stat, critical_value, alpha=0.05, 
                                  test_type='two-tailed'):
        """Visualize hypothesis test"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        
        ax.plot(x, y, 'b-', linewidth=2, label='Null Distribution')
        ax.fill_between(x, y, alpha=0.3)
        
        if test_type == 'two-tailed':
            # Rejection regions
            left_crit = -critical_value
            right_crit = critical_value
            
            ax.fill_between(x[x <= left_crit], 0, y[x <= left_crit], 
                           color='red', alpha=0.5, label=f'Rejection Region (α/2={alpha/2})')
            ax.fill_between(x[x >= right_crit], 0, y[x >= right_crit], 
                           color='red', alpha=0.5)
            
            ax.axvline(left_crit, color='red', linestyle='--', linewidth=2)
            ax.axvline(right_crit, color='red', linestyle='--', linewidth=2)
            
        elif test_type == 'right-tailed':
            ax.fill_between(x[x >= critical_value], 0, y[x >= critical_value], 
                           color='red', alpha=0.5, label=f'Rejection Region (α={alpha})')
            ax.axvline(critical_value, color='red', linestyle='--', linewidth=2)
        
        # Mark test statistic
        ax.axvline(test_stat, color='green', linestyle='-', linewidth=3, 
                  label=f'Test Statistic = {test_stat:.2f}')
        
        ax.set_xlabel('Test Statistic')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Hypothesis Test Visualization ({test_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class OneSampleTests:
    """
    Tests for a single sample
    """
    
    @staticmethod
    def z_test(sample_mean, pop_mean, pop_std, n, alpha=0.05, alternative='two-sided'):
        """
        One-Sample Z-Test
        Use when: Population std is known, large sample (n ≥ 30)
        H₀: μ = μ₀
        """
        print("\n" + "="*60)
        print("ONE-SAMPLE Z-TEST")
        print("="*60)
        
        # Calculate z-statistic
        se = pop_std / np.sqrt(n)
        z_stat = (sample_mean - pop_mean) / se
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:  # less
            p_value = stats.norm.cdf(z_stat)
        
        print(f"\nH₀: μ = {pop_mean}")
        print(f"H₁: μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        print(f"\nSample mean: {sample_mean}")
        print(f"Population mean (H₀): {pop_mean}")
        print(f"Population std: {pop_std}")
        print(f"Sample size: {n}")
        print(f"Standard error: {se:.4f}")
        print(f"\nZ-statistic: {z_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significance level: {alpha}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print(f"Conclusion: Significant evidence that μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print(f"Conclusion: Insufficient evidence that μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        
        return z_stat, p_value
    
    @staticmethod
    def t_test(data, pop_mean, alpha=0.05, alternative='two-sided'):
        """
        One-Sample T-Test
        Use when: Population std unknown, any sample size
        H₀: μ = μ₀
        """
        print("\n" + "="*60)
        print("ONE-SAMPLE T-TEST")
        print("="*60)
        
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        se = sample_std / np.sqrt(n)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(data, pop_mean, alternative=alternative)
        
        # Critical value
        if alternative == 'two-sided':
            critical_value = stats.t.ppf(1 - alpha/2, n-1)
        else:
            critical_value = stats.t.ppf(1 - alpha, n-1)
        
        print(f"\nH₀: μ = {pop_mean}")
        print(f"H₁: μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        print(f"\nSample size: {n}")
        print(f"Sample mean: {sample_mean:.4f}")
        print(f"Sample std: {sample_std:.4f}")
        print(f"Standard error: {se:.4f}")
        print(f"Degrees of freedom: {n-1}")
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"Critical value: ±{critical_value:.4f}" if alternative == 'two-sided' else f"{critical_value:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print(f"Conclusion: Significant evidence that μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print(f"Conclusion: Insufficient evidence that μ {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {pop_mean}")
        
        return t_stat, p_value


class TwoSampleTests:
    """
    Tests comparing two samples
    """
    
    @staticmethod
    def independent_t_test(data1, data2, alpha=0.05, equal_var=True):
        """
        Independent Two-Sample T-Test
        Use when: Comparing means of two independent groups
        H₀: μ₁ = μ₂
        """
        print("\n" + "="*60)
        print("INDEPENDENT TWO-SAMPLE T-TEST")
        print("="*60)
        
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        print(f"\nH₀: μ₁ = μ₂ (no difference between groups)")
        print(f"H₁: μ₁ ≠ μ₂ (groups are different)")
        
        print(f"\nGroup 1: n={n1}, mean={mean1:.2f}, std={std1:.2f}")
        print(f"Group 2: n={n2}, mean={mean2:.2f}, std={std2:.2f}")
        print(f"Difference in means: {mean1 - mean2:.2f}")
        
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print("Conclusion: Significant difference between groups")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print("Conclusion: No significant difference between groups")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print("  → Small effect")
        elif abs(cohens_d) < 0.5:
            print("  → Medium effect")
        else:
            print("  → Large effect")
        
        return t_stat, p_value
    
    @staticmethod
    def paired_t_test(before, after, alpha=0.05):
        """
        Paired T-Test
        Use when: Comparing same subjects before/after treatment
        H₀: μ_diff = 0
        """
        print("\n" + "="*60)
        print("PAIRED T-TEST")
        print("="*60)
        
        differences = np.array(after) - np.array(before)
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(after, before)
        
        print(f"\nH₀: μ_diff = 0 (no change)")
        print(f"H₁: μ_diff ≠ 0 (significant change)")
        
        print(f"\nPairs: {n}")
        print(f"Mean before: {np.mean(before):.2f}")
        print(f"Mean after: {np.mean(after):.2f}")
        print(f"Mean difference: {mean_diff:.2f}")
        print(f"Std of differences: {std_diff:.2f}")
        
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print("Conclusion: Significant change detected")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print("Conclusion: No significant change detected")
        
        return t_stat, p_value


class ProportionTests:
    """
    Tests for proportions
    """
    
    @staticmethod
    def one_proportion_test(successes, n, p0, alpha=0.05, alternative='two-sided'):
        """
        One-Proportion Z-Test
        Use when: Testing if sample proportion differs from known proportion
        H₀: p = p₀
        """
        print("\n" + "="*60)
        print("ONE-PROPORTION Z-TEST")
        print("="*60)
        
        p_hat = successes / n
        
        # Calculate z-statistic
        se = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (p_hat - p0) / se
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
        
        print(f"\nH₀: p = {p0}")
        print(f"H₁: p {'≠' if alternative == 'two-sided' else '>' if alternative == 'greater' else '<'} {p0}")
        
        print(f"\nSample size: {n}")
        print(f"Successes: {successes}")
        print(f"Sample proportion: {p_hat:.4f}")
        print(f"Null proportion: {p0}")
        
        print(f"\nZ-statistic: {z_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print(f"Conclusion: Proportion is significantly {'different from' if alternative == 'two-sided' else 'greater than' if alternative == 'greater' else 'less than'} {p0}")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print(f"Conclusion: Insufficient evidence that proportion {'differs from' if alternative == 'two-sided' else 'is greater than' if alternative == 'greater' else 'is less than'} {p0}")
        
        return z_stat, p_value


class ChiSquareTests:
    """
    Chi-Square Tests
    """
    
    @staticmethod
    def goodness_of_fit(observed, expected, alpha=0.05):
        """
        Chi-Square Goodness of Fit Test
        Use when: Testing if observed frequencies match expected distribution
        H₀: Data follows expected distribution
        """
        print("\n" + "="*60)
        print("CHI-SQUARE GOODNESS OF FIT TEST")
        print("="*60)
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        df = len(observed) - 1
        
        print(f"\nH₀: Data follows expected distribution")
        print(f"H₁: Data does not follow expected distribution")
        
        print(f"\nObserved: {observed}")
        print(f"Expected: {expected}")
        print(f"Degrees of freedom: {df}")
        
        print(f"\nχ² statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print("Conclusion: Data does NOT follow expected distribution")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print("Conclusion: Data is consistent with expected distribution")
        
        return chi2_stat, p_value
    
    @staticmethod
    def independence_test(contingency_table, alpha=0.05):
        """
        Chi-Square Test of Independence
        Use when: Testing if two categorical variables are independent
        H₀: Variables are independent
        """
        print("\n" + "="*60)
        print("CHI-SQUARE TEST OF INDEPENDENCE")
        print("="*60)
        
        chi2_stat, p_value, df, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\nH₀: Variables are independent")
        print(f"H₁: Variables are associated")
        
        print(f"\nObserved frequencies:")
        print(pd.DataFrame(contingency_table))
        
        print(f"\nExpected frequencies:")
        print(pd.DataFrame(expected))
        
        print(f"\nχ² statistic: {chi2_stat:.4f}")
        print(f"Degrees of freedom: {df}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print("Conclusion: Variables are associated (not independent)")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print("Conclusion: Variables appear to be independent")
        
        return chi2_stat, p_value


# ==================== PRACTICAL EXAMPLES ====================

def example_1_one_sample():
    """Example 1: One-sample t-test"""
    print("\n" + "="*60)
    print("EXAMPLE 1: One-Sample T-Test - Coffee Shop Wait Times")
    print("="*60)
    
    print("\nScenario: Coffee shop claims average wait time is 5 minutes")
    print("You measure wait times for 20 customers")
    
    # Sample data
    np.random.seed(42)
    wait_times = np.random.normal(5.8, 1.5, 20)
    
    print(f"\nSample data (minutes): {wait_times[:5]}... (showing first 5)")
    
    OneSampleTests.t_test(wait_times, pop_mean=5, alpha=0.05)


def example_2_two_sample():
    """Example 2: Independent two-sample t-test"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Two-Sample T-Test - A/B Testing")
    print("="*60)
    
    print("\nScenario: Testing two website designs")
    print("Measuring time spent on page (seconds)")
    
    np.random.seed(42)
    design_a = np.random.normal(45, 10, 30)
    design_b = np.random.normal(52, 12, 30)
    
    TwoSampleTests.independent_t_test(design_a, design_b, alpha=0.05)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([design_a, design_b], labels=['Design A', 'Design B'])
    ax.set_ylabel('Time on Page (seconds)')
    ax.set_title('A/B Test: Website Design Comparison')
    ax.grid(True, alpha=0.3)
    plt.savefig('/vercel/sandbox/ab_test_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_3_paired():
    """Example 3: Paired t-test"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Paired T-Test - Training Program Effectiveness")
    print("="*60)
    
    print("\nScenario: Measuring employee productivity before/after training")
    
    np.random.seed(42)
    before = np.random.normal(75, 10, 15)
    after = before + np.random.normal(5, 3, 15)  # Improvement
    
    TwoSampleTests.paired_t_test(before, after, alpha=0.05)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before vs After
    ax1.plot(range(len(before)), before, 'o-', label='Before', linewidth=2)
    ax1.plot(range(len(after)), after, 's-', label='After', linewidth=2)
    ax1.set_xlabel('Employee')
    ax1.set_ylabel('Productivity Score')
    ax1.set_title('Before vs After Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Differences
    differences = after - before
    ax2.bar(range(len(differences)), differences, color='green', alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Employee')
    ax2.set_ylabel('Change in Productivity')
    ax2.set_title('Improvement After Training')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/vercel/sandbox/paired_test_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_4_proportion():
    """Example 4: Proportion test"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Proportion Test - Quality Control")
    print("="*60)
    
    print("\nScenario: Factory claims 95% of products pass quality check")
    print("In sample of 200 products, 182 passed")
    
    ProportionTests.one_proportion_test(
        successes=182, n=200, p0=0.95, alpha=0.05, alternative='less'
    )


def example_5_chi_square():
    """Example 5: Chi-square test"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Chi-Square Test - Dice Fairness")
    print("="*60)
    
    print("\nScenario: Testing if a die is fair")
    print("Rolled 120 times")
    
    observed = [18, 22, 19, 21, 20, 20]  # Observed frequencies
    expected = [20, 20, 20, 20, 20, 20]  # Expected if fair
    
    ChiSquareTests.goodness_of_fit(observed, expected, alpha=0.05)


def example_6_independence():
    """Example 6: Chi-square independence test"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Chi-Square Independence - Marketing Campaign")
    print("="*60)
    
    print("\nScenario: Is purchase behavior independent of age group?")
    
    # Contingency table: rows=age groups, cols=purchased/not purchased
    data = np.array([
        [30, 70],  # Young: 30 purchased, 70 didn't
        [50, 50],  # Middle: 50 purchased, 50 didn't
        [20, 80]   # Senior: 20 purchased, 80 didn't
    ])
    
    ChiSquareTests.independence_test(data, alpha=0.05)


def example_7_power_analysis():
    """Example 7: Statistical Power"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Understanding Statistical Power")
    print("="*60)
    
    print("\nPower = Probability of detecting an effect when it exists")
    print("Affected by: sample size, effect size, significance level")
    
    # Simulate power analysis
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
    sample_sizes = range(10, 201, 10)
    alpha = 0.05
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for effect_size in effect_sizes:
        powers = []
        for n in sample_sizes:
            # Simulate power
            noncentrality = effect_size * np.sqrt(n)
            critical_value = stats.norm.ppf(1 - alpha/2)
            power = 1 - stats.norm.cdf(critical_value - noncentrality)
            powers.append(power)
        
        ax.plot(sample_sizes, powers, linewidth=2, 
               label=f'Effect size = {effect_size}')
    
    ax.axhline(0.8, color='red', linestyle='--', linewidth=2, 
              label='Target power = 0.8')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Statistical Power')
    ax.set_title('Power Analysis: Effect of Sample Size and Effect Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.savefig('/vercel/sandbox/power_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'power_analysis.png'")
    print("\n📊 Key insight: Larger samples and larger effects → higher power")
    plt.show()


# ==================== INTERACTIVE EXERCISES ====================

def exercise_1():
    """Exercise: Hypothesis test"""
    print("\n" + "="*60)
    print("EXERCISE 1: Conduct a Hypothesis Test")
    print("="*60)
    
    print("\nScenario: Restaurant claims average meal costs $25")
    print("Sample of 25 meals: mean=$27, std=$5")
    print("\nQuestion: Is the actual average significantly different?")
    
    print("\n--- SOLUTION ---")
    
    # Generate sample data matching these statistics
    np.random.seed(42)
    sample = np.random.normal(27, 5, 25)
    
    OneSampleTests.t_test(sample, pop_mean=25, alpha=0.05)


def exercise_2():
    """Exercise: Choose correct test"""
    print("\n" + "="*60)
    print("EXERCISE 2: Choose the Correct Test")
    print("="*60)
    
    scenarios = [
        ("Comparing heights of men vs women", "Independent two-sample t-test"),
        ("Testing if a coin is fair", "Chi-square goodness of fit"),
        ("Weight before vs after diet", "Paired t-test"),
        ("Is average IQ different from 100?", "One-sample t-test"),
        ("Is gender independent of major?", "Chi-square independence test"),
        ("Is conversion rate > 10%?", "One-proportion z-test")
    ]
    
    print("\nMatch each scenario with the correct test:\n")
    for i, (scenario, test) in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    print("\n--- ANSWERS ---")
    for i, (scenario, test) in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
        print(f"   → {test}\n")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("WELCOME TO MODULE 4: HYPOTHESIS TESTING")
    print("🔬 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Understand null and alternative hypotheses")
    print("2. Interpret p-values and significance levels")
    print("3. Conduct one-sample and two-sample tests")
    print("4. Perform chi-square tests")
    print("5. Understand Type I and Type II errors")
    print("6. Calculate statistical power")
    
    # Explain fundamentals
    HypothesisTestingBasics.explain_concepts()
    
    # Run examples
    example_1_one_sample()
    example_2_two_sample()
    example_3_paired()
    example_4_proportion()
    example_5_chi_square()
    example_6_independence()
    example_7_power_analysis()
    
    # Run exercises
    exercise_1()
    exercise_2()
    
    print("\n" + "="*60)
    print("✅ MODULE 4 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• p-value < α → Reject H₀ (statistically significant)")
    print("• Choose test based on data type and research question")
    print("• Independent samples: two-sample t-test")
    print("• Paired data: paired t-test")
    print("• Categorical data: chi-square tests")
    print("• Larger samples → more power to detect effects")
    print("• Statistical significance ≠ practical significance")
    print("\n➡️  Next: Module 5 - Regression Analysis")
