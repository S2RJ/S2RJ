"""
Module 3: Probability Distributions - Modeling Random Phenomena
================================================================
Learn the most important probability distributions used in data science.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import comb

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


# ==================== DISCRETE DISTRIBUTIONS ====================

class BinomialDistribution:
    """
    Binomial Distribution: Number of successes in n independent trials
    Parameters: n (trials), p (success probability)
    Use: Counting successes (coin flips, quality control, A/B testing)
    """
    
    def __init__(self, n, p):
        self.n = n  # number of trials
        self.p = p  # probability of success
        self.dist = stats.binom(n, p)
    
    def explain(self):
        print(f"\n{'='*60}")
        print("BINOMIAL DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Parameters: n={self.n} trials, p={self.p} success probability")
        print(f"\nMean (Expected): μ = n×p = {self.n}×{self.p} = {self.dist.mean():.2f}")
        print(f"Variance: σ² = n×p×(1-p) = {self.n}×{self.p}×{1-self.p} = {self.dist.var():.2f}")
        print(f"Std Dev: σ = {self.dist.std():.2f}")
        
    def probability(self, k):
        """P(X = k): Probability of exactly k successes"""
        prob = self.dist.pmf(k)
        print(f"\nP(X = {k}) = {prob:.4f}")
        return prob
    
    def cumulative(self, k):
        """P(X ≤ k): Probability of k or fewer successes"""
        prob = self.dist.cdf(k)
        print(f"P(X ≤ {k}) = {prob:.4f}")
        return prob
    
    def visualize(self):
        """Visualize the distribution"""
        x = np.arange(0, self.n + 1)
        pmf = self.dist.pmf(x)
        cdf = self.dist.cdf(x)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PMF
        ax1.bar(x, pmf, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(self.dist.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {self.dist.mean():.1f}')
        ax1.set_xlabel('Number of Successes (k)')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Binomial PMF (n={self.n}, p={self.p})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        ax2.step(x, cdf, where='post', color='darkgreen', linewidth=2)
        ax2.set_xlabel('Number of Successes (k)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'Binomial CDF (n={self.n}, p={self.p})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class PoissonDistribution:
    """
    Poisson Distribution: Number of events in fixed interval
    Parameter: λ (lambda) - average rate
    Use: Rare events (website visits, calls per hour, defects)
    """
    
    def __init__(self, lambda_):
        self.lambda_ = lambda_  # average rate
        self.dist = stats.poisson(lambda_)
    
    def explain(self):
        print(f"\n{'='*60}")
        print("POISSON DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Parameter: λ={self.lambda_} (average rate)")
        print(f"\nMean: μ = λ = {self.lambda_}")
        print(f"Variance: σ² = λ = {self.lambda_}")
        print(f"Std Dev: σ = √λ = {np.sqrt(self.lambda_):.2f}")
        print("\nUse when:")
        print("• Events occur independently")
        print("• Average rate is constant")
        print("• Events are rare")
        
    def probability(self, k):
        """P(X = k): Probability of exactly k events"""
        prob = self.dist.pmf(k)
        print(f"\nP(X = {k}) = {prob:.4f}")
        return prob
    
    def visualize(self, max_k=None):
        """Visualize the distribution"""
        if max_k is None:
            max_k = int(self.lambda_ + 4 * np.sqrt(self.lambda_))
        
        x = np.arange(0, max_k + 1)
        pmf = self.dist.pmf(x)
        
        plt.figure(figsize=(10, 6))
        plt.bar(x, pmf, color='coral', alpha=0.7, edgecolor='black')
        plt.axvline(self.lambda_, color='red', linestyle='--', 
                   linewidth=2, label=f'λ = {self.lambda_}')
        plt.xlabel('Number of Events (k)')
        plt.ylabel('Probability')
        plt.title(f'Poisson Distribution (λ={self.lambda_})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()


# ==================== CONTINUOUS DISTRIBUTIONS ====================

class NormalDistribution:
    """
    Normal (Gaussian) Distribution: The bell curve
    Parameters: μ (mean), σ (standard deviation)
    Use: Most important distribution - Central Limit Theorem
    """
    
    def __init__(self, mu, sigma):
        self.mu = mu  # mean
        self.sigma = sigma  # standard deviation
        self.dist = stats.norm(mu, sigma)
    
    def explain(self):
        print(f"\n{'='*60}")
        print("NORMAL DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Parameters: μ={self.mu} (mean), σ={self.sigma} (std dev)")
        print(f"\nProperties:")
        print(f"• Symmetric around mean")
        print(f"• 68% of data within μ ± σ")
        print(f"• 95% of data within μ ± 2σ")
        print(f"• 99.7% of data within μ ± 3σ (Empirical Rule)")
        
    def probability_range(self, a, b):
        """P(a < X < b): Probability between a and b"""
        prob = self.dist.cdf(b) - self.dist.cdf(a)
        print(f"\nP({a} < X < {b}) = {prob:.4f}")
        return prob
    
    def z_score(self, x):
        """
        Z-score: Number of standard deviations from mean
        z = (x - μ) / σ
        """
        z = (x - self.mu) / self.sigma
        print(f"\nZ-score for x={x}: z = ({x} - {self.mu}) / {self.sigma} = {z:.2f}")
        print(f"Interpretation: {x} is {abs(z):.2f} std devs {'above' if z > 0 else 'below'} mean")
        return z
    
    def percentile(self, p):
        """Find value at given percentile"""
        value = self.dist.ppf(p/100)
        print(f"\n{p}th percentile: {value:.2f}")
        return value
    
    def visualize(self):
        """Visualize the distribution"""
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        pdf = self.dist.pdf(x)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot PDF
        ax.plot(x, pdf, 'b-', linewidth=2, label='PDF')
        ax.fill_between(x, pdf, alpha=0.3)
        
        # Mark mean
        ax.axvline(self.mu, color='red', linestyle='--', linewidth=2, label=f'μ = {self.mu}')
        
        # Mark standard deviations
        colors = ['green', 'orange', 'purple']
        for i, color in enumerate(colors, 1):
            ax.axvline(self.mu + i*self.sigma, color=color, linestyle=':', alpha=0.7)
            ax.axvline(self.mu - i*self.sigma, color=color, linestyle=':', alpha=0.7)
            if i == 1:
                ax.text(self.mu + i*self.sigma, max(pdf)*0.5, f'+{i}σ', fontsize=10)
        
        # Shade 68-95-99.7 regions
        x_1sigma = x[(x >= self.mu - self.sigma) & (x <= self.mu + self.sigma)]
        ax.fill_between(x_1sigma, 0, self.dist.pdf(x_1sigma), 
                       alpha=0.3, color='green', label='68% (±1σ)')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Normal Distribution (μ={self.mu}, σ={self.sigma})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class ExponentialDistribution:
    """
    Exponential Distribution: Time between events
    Parameter: λ (rate)
    Use: Waiting times, lifetimes, time between arrivals
    """
    
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.dist = stats.expon(scale=1/lambda_)
    
    def explain(self):
        print(f"\n{'='*60}")
        print("EXPONENTIAL DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Parameter: λ={self.lambda_} (rate)")
        print(f"\nMean: μ = 1/λ = {1/self.lambda_:.2f}")
        print(f"Variance: σ² = 1/λ² = {1/self.lambda_**2:.2f}")
        print(f"Std Dev: σ = 1/λ = {1/self.lambda_:.2f}")
        print("\nMemoryless property: P(X > s+t | X > s) = P(X > t)")
        
    def probability_greater(self, t):
        """P(X > t): Probability of waiting more than t"""
        prob = 1 - self.dist.cdf(t)
        print(f"\nP(X > {t}) = {prob:.4f}")
        return prob
    
    def visualize(self):
        """Visualize the distribution"""
        x = np.linspace(0, 5/self.lambda_, 1000)
        pdf = self.dist.pdf(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf, 'b-', linewidth=2)
        plt.fill_between(x, pdf, alpha=0.3)
        plt.axvline(1/self.lambda_, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {1/self.lambda_:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Probability Density')
        plt.title(f'Exponential Distribution (λ={self.lambda_})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()


class UniformDistribution:
    """
    Uniform Distribution: All values equally likely
    Parameters: a (min), b (max)
    Use: Random number generation, modeling complete uncertainty
    """
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.dist = stats.uniform(a, b - a)
    
    def explain(self):
        print(f"\n{'='*60}")
        print("UNIFORM DISTRIBUTION")
        print(f"{'='*60}")
        print(f"Parameters: a={self.a} (min), b={self.b} (max)")
        print(f"\nMean: μ = (a+b)/2 = {(self.a + self.b)/2:.2f}")
        print(f"Variance: σ² = (b-a)²/12 = {(self.b - self.a)**2/12:.2f}")
        print(f"All values in [{self.a}, {self.b}] equally likely")
    
    def visualize(self):
        """Visualize the distribution"""
        x = np.linspace(self.a - 1, self.b + 1, 1000)
        pdf = self.dist.pdf(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf, 'b-', linewidth=2)
        plt.fill_between(x, pdf, alpha=0.3)
        plt.axvline((self.a + self.b)/2, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {(self.a + self.b)/2:.2f}')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.title(f'Uniform Distribution [{self.a}, {self.b}]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()


# ==================== PRACTICAL EXAMPLES ====================

def example_1_binomial():
    """Example 1: Binomial - Quality Control"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Binomial Distribution - Quality Control")
    print("="*60)
    
    print("\nScenario: Manufacturing produces 5% defective items")
    print("Inspect 20 items. What's probability of finding:")
    
    n, p = 20, 0.05
    binom = BinomialDistribution(n, p)
    binom.explain()
    
    print("\n1. Exactly 2 defective items?")
    binom.probability(2)
    
    print("\n2. At most 1 defective item?")
    binom.cumulative(1)
    
    print("\n3. More than 2 defective items?")
    prob_more_than_2 = 1 - binom.cumulative(2)
    print(f"P(X > 2) = 1 - P(X ≤ 2) = {prob_more_than_2:.4f}")
    
    fig = binom.visualize()
    plt.savefig('/vercel/sandbox/binomial_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_2_poisson():
    """Example 2: Poisson - Website Traffic"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Poisson Distribution - Website Traffic")
    print("="*60)
    
    print("\nScenario: Website averages 3 visitors per minute")
    print("What's probability of:")
    
    lambda_ = 3
    poisson = PoissonDistribution(lambda_)
    poisson.explain()
    
    print("\n1. Exactly 5 visitors in next minute?")
    poisson.probability(5)
    
    print("\n2. No visitors in next minute?")
    poisson.probability(0)
    
    print("\n3. More than 5 visitors?")
    prob_more_5 = 1 - poisson.dist.cdf(5)
    print(f"P(X > 5) = {prob_more_5:.4f}")
    
    fig = poisson.visualize()
    plt.savefig('/vercel/sandbox/poisson_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_3_normal():
    """Example 3: Normal - IQ Scores"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Normal Distribution - IQ Scores")
    print("="*60)
    
    print("\nScenario: IQ scores ~ N(100, 15)")
    print("Mean = 100, Std Dev = 15")
    
    mu, sigma = 100, 15
    normal = NormalDistribution(mu, sigma)
    normal.explain()
    
    print("\n1. What's probability of IQ between 85 and 115?")
    normal.probability_range(85, 115)
    
    print("\n2. What's probability of IQ above 130?")
    prob_above_130 = 1 - normal.dist.cdf(130)
    print(f"P(X > 130) = {prob_above_130:.4f}")
    
    print("\n3. What IQ score is at 90th percentile?")
    normal.percentile(90)
    
    print("\n4. Convert IQ of 130 to z-score:")
    normal.z_score(130)
    
    fig = normal.visualize()
    plt.savefig('/vercel/sandbox/normal_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_4_exponential():
    """Example 4: Exponential - Customer Service"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Exponential Distribution - Wait Times")
    print("="*60)
    
    print("\nScenario: Average 4 customers per hour (λ=4)")
    print("Time between customers follows Exponential(4)")
    
    lambda_ = 4
    exp = ExponentialDistribution(lambda_)
    exp.explain()
    
    print("\n1. Probability of waiting more than 30 minutes (0.5 hours)?")
    exp.probability_greater(0.5)
    
    print("\n2. Probability of waiting less than 15 minutes (0.25 hours)?")
    prob_less_15 = exp.dist.cdf(0.25)
    print(f"P(X < 0.25) = {prob_less_15:.4f}")
    
    fig = exp.visualize()
    plt.savefig('/vercel/sandbox/exponential_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_5_clt():
    """Example 5: Central Limit Theorem"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Central Limit Theorem (CLT)")
    print("="*60)
    
    print("\nCLT: Sample means approach normal distribution")
    print("regardless of original distribution!")
    
    print("\nDemonstration: Rolling dice")
    print("Single die: Uniform distribution")
    print("Average of many dice: Normal distribution")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Central Limit Theorem: Dice Rolling', fontsize=16, fontweight='bold')
    
    sample_sizes = [1, 2, 5, 10, 30, 100]
    
    for idx, n in enumerate(sample_sizes):
        row = idx // 3
        col = idx % 3
        
        # Simulate rolling n dice, 10000 times
        samples = np.random.randint(1, 7, size=(10000, n))
        means = samples.mean(axis=1)
        
        axes[row, col].hist(means, bins=30, density=True, alpha=0.7, 
                           edgecolor='black', color='skyblue')
        axes[row, col].set_title(f'n = {n} dice')
        axes[row, col].set_xlabel('Average Value')
        axes[row, col].set_ylabel('Density')
        
        # Overlay normal curve
        mu = means.mean()
        sigma = means.std()
        x = np.linspace(means.min(), means.max(), 100)
        axes[row, col].plot(x, stats.norm.pdf(x, mu, sigma), 
                           'r-', linewidth=2, label='Normal fit')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/vercel/sandbox/central_limit_theorem.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'central_limit_theorem.png'")
    print("\n📊 Observation: As n increases, distribution becomes more normal!")
    plt.show()


def example_6_compare_distributions():
    """Example 6: Comparing Distributions"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Comparing Different Distributions")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Common Probability Distributions', fontsize=16, fontweight='bold')
    
    # Binomial
    n, p = 20, 0.3
    x_binom = np.arange(0, n+1)
    axes[0, 0].bar(x_binom, stats.binom.pmf(x_binom, n, p), 
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Binomial(n={n}, p={p})')
    axes[0, 0].set_xlabel('k')
    axes[0, 0].set_ylabel('P(X=k)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Poisson
    lambda_ = 5
    x_poisson = np.arange(0, 20)
    axes[0, 1].bar(x_poisson, stats.poisson.pmf(x_poisson, lambda_), 
                   color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'Poisson(λ={lambda_})')
    axes[0, 1].set_xlabel('k')
    axes[0, 1].set_ylabel('P(X=k)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Normal
    mu, sigma = 0, 1
    x_normal = np.linspace(-4, 4, 1000)
    axes[1, 0].plot(x_normal, stats.norm.pdf(x_normal, mu, sigma), 
                   'b-', linewidth=2)
    axes[1, 0].fill_between(x_normal, stats.norm.pdf(x_normal, mu, sigma), alpha=0.3)
    axes[1, 0].set_title(f'Normal(μ={mu}, σ={sigma})')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Exponential
    lambda_exp = 1
    x_exp = np.linspace(0, 5, 1000)
    axes[1, 1].plot(x_exp, stats.expon.pdf(x_exp, scale=1/lambda_exp), 
                   'g-', linewidth=2)
    axes[1, 1].fill_between(x_exp, stats.expon.pdf(x_exp, scale=1/lambda_exp), alpha=0.3)
    axes[1, 1].set_title(f'Exponential(λ={lambda_exp})')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('f(x)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/vercel/sandbox/distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison saved as 'distribution_comparison.png'")
    plt.show()


# ==================== INTERACTIVE EXERCISES ====================

def exercise_1():
    """Exercise: Binomial probability"""
    print("\n" + "="*60)
    print("EXERCISE 1: Binomial Distribution")
    print("="*60)
    
    print("\nScenario: 60% of customers buy after demo")
    print("You give 10 demos today")
    print("\nQuestions:")
    print("1. Expected number of sales?")
    print("2. P(exactly 6 sales)?")
    print("3. P(at least 8 sales)?")
    
    print("\n--- SOLUTIONS ---")
    n, p = 10, 0.6
    binom = BinomialDistribution(n, p)
    
    print("\n1. Expected sales:")
    print(f"E(X) = n×p = {n}×{p} = {n*p}")
    
    print("\n2. P(X = 6):")
    binom.probability(6)
    
    print("\n3. P(X ≥ 8):")
    prob = 1 - binom.cumulative(7)
    print(f"P(X ≥ 8) = 1 - P(X ≤ 7) = {prob:.4f}")


def exercise_2():
    """Exercise: Normal distribution"""
    print("\n" + "="*60)
    print("EXERCISE 2: Normal Distribution")
    print("="*60)
    
    print("\nScenario: Heights ~ N(170, 10) cm")
    print("\nQuestions:")
    print("1. P(height > 180)?")
    print("2. P(160 < height < 180)?")
    print("3. What height is at 25th percentile?")
    
    print("\n--- SOLUTIONS ---")
    normal = NormalDistribution(170, 10)
    
    print("\n1. P(X > 180):")
    prob1 = 1 - normal.dist.cdf(180)
    print(f"P(X > 180) = {prob1:.4f}")
    
    print("\n2. P(160 < X < 180):")
    normal.probability_range(160, 180)
    
    print("\n3. 25th percentile:")
    normal.percentile(25)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "📊 " * 20)
    print("WELCOME TO MODULE 3: PROBABILITY DISTRIBUTIONS")
    print("📊 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Master discrete distributions (Binomial, Poisson)")
    print("2. Master continuous distributions (Normal, Exponential, Uniform)")
    print("3. Understand when to use each distribution")
    print("4. Apply Central Limit Theorem")
    print("5. Calculate probabilities and percentiles")
    
    # Run examples
    example_1_binomial()
    example_2_poisson()
    example_3_normal()
    example_4_exponential()
    example_5_clt()
    example_6_compare_distributions()
    
    # Run exercises
    exercise_1()
    exercise_2()
    
    print("\n" + "="*60)
    print("✅ MODULE 3 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Binomial: Count successes in fixed trials")
    print("• Poisson: Count rare events in interval")
    print("• Normal: Most important - CLT makes it universal")
    print("• Exponential: Time between events")
    print("• 68-95-99.7 rule for normal distribution")
    print("• Z-scores standardize normal distributions")
    print("\n➡️  Next: Module 4 - Hypothesis Testing")
