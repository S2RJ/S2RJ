"""
Module 1: Descriptive Statistics - The Foundation
==================================================
Learn how to summarize and describe data using measures of central tendency,
variability, and visualization techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DescriptiveStatistics:
    """
    A comprehensive class for understanding descriptive statistics.
    """
    
    def __init__(self, data):
        """Initialize with data (list, array, or pandas Series)"""
        self.data = np.array(data)
        self.n = len(self.data)
    
    # ==================== MEASURES OF CENTRAL TENDENCY ====================
    
    def mean(self):
        """
        Mean (Average): Sum of all values divided by count
        Formula: μ = Σx / n
        Use: Best for symmetric distributions without outliers
        """
        result = np.sum(self.data) / self.n
        print(f"Mean: {result:.2f}")
        print(f"Interpretation: The average value in the dataset")
        return result
    
    def median(self):
        """
        Median: Middle value when data is sorted
        Use: Better than mean for skewed distributions or with outliers
        """
        sorted_data = np.sort(self.data)
        if self.n % 2 == 0:
            result = (sorted_data[self.n//2 - 1] + sorted_data[self.n//2]) / 2
        else:
            result = sorted_data[self.n//2]
        print(f"Median: {result:.2f}")
        print(f"Interpretation: 50% of values are below this point")
        return result
    
    def mode(self):
        """
        Mode: Most frequently occurring value(s)
        Use: Best for categorical data or finding most common value
        """
        values, counts = np.unique(self.data, return_counts=True)
        max_count = np.max(counts)
        modes = values[counts == max_count]
        print(f"Mode: {modes}")
        print(f"Frequency: {max_count} times")
        return modes
    
    # ==================== MEASURES OF VARIABILITY ====================
    
    def variance(self, sample=True):
        """
        Variance: Average squared deviation from mean
        Formula: σ² = Σ(x - μ)² / n  (population)
                 s² = Σ(x - μ)² / (n-1)  (sample)
        Use: Measures spread of data
        """
        mean_val = np.mean(self.data)
        squared_diff = (self.data - mean_val) ** 2
        divisor = self.n - 1 if sample else self.n
        result = np.sum(squared_diff) / divisor
        var_type = "Sample" if sample else "Population"
        print(f"{var_type} Variance: {result:.2f}")
        print(f"Interpretation: Higher variance = more spread out data")
        return result
    
    def standard_deviation(self, sample=True):
        """
        Standard Deviation: Square root of variance
        Formula: σ = √(variance)
        Use: Measures spread in same units as original data
        """
        result = np.sqrt(self.variance(sample))
        print(f"Standard Deviation: {result:.2f}")
        print(f"Interpretation: On average, values deviate by {result:.2f} from mean")
        return result
    
    def range_stat(self):
        """
        Range: Difference between maximum and minimum
        Use: Quick measure of spread, sensitive to outliers
        """
        result = np.max(self.data) - np.min(self.data)
        print(f"Range: {result:.2f} (from {np.min(self.data):.2f} to {np.max(self.data):.2f})")
        return result
    
    def iqr(self):
        """
        Interquartile Range (IQR): Q3 - Q1
        Use: Robust measure of spread, not affected by outliers
        """
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        result = q3 - q1
        print(f"IQR: {result:.2f} (Q1={q1:.2f}, Q3={q3:.2f})")
        print(f"Interpretation: Middle 50% of data spans {result:.2f} units")
        return result
    
    # ==================== MEASURES OF SHAPE ====================
    
    def skewness(self):
        """
        Skewness: Measure of asymmetry
        - Negative: Left-skewed (tail on left)
        - Zero: Symmetric
        - Positive: Right-skewed (tail on right)
        """
        result = stats.skew(self.data)
        print(f"Skewness: {result:.2f}")
        if result < -0.5:
            print("Interpretation: Left-skewed (mean < median)")
        elif result > 0.5:
            print("Interpretation: Right-skewed (mean > median)")
        else:
            print("Interpretation: Approximately symmetric")
        return result
    
    def kurtosis(self):
        """
        Kurtosis: Measure of "tailedness"
        - Negative: Flatter than normal (platykurtic)
        - Zero: Normal distribution (mesokurtic)
        - Positive: More peaked than normal (leptokurtic)
        """
        result = stats.kurtosis(self.data)
        print(f"Kurtosis: {result:.2f}")
        if result > 0:
            print("Interpretation: Heavy tails, more outliers than normal")
        elif result < 0:
            print("Interpretation: Light tails, fewer outliers than normal")
        else:
            print("Interpretation: Similar to normal distribution")
        return result
    
    # ==================== PERCENTILES & QUANTILES ====================
    
    def percentiles(self, percentiles=[25, 50, 75, 90, 95, 99]):
        """
        Percentiles: Values below which a percentage of data falls
        Use: Understanding data distribution
        """
        print("\nPercentiles:")
        results = {}
        for p in percentiles:
            value = np.percentile(self.data, p)
            results[p] = value
            print(f"  {p}th percentile: {value:.2f} ({p}% of data is below this)")
        return results
    
    # ==================== COMPREHENSIVE SUMMARY ====================
    
    def summary(self):
        """
        Complete statistical summary of the data
        """
        print("=" * 60)
        print("DESCRIPTIVE STATISTICS SUMMARY")
        print("=" * 60)
        print(f"\nDataset size: {self.n} observations")
        print(f"\n--- Central Tendency ---")
        mean_val = self.mean()
        median_val = self.median()
        mode_val = self.mode()
        
        print(f"\n--- Variability ---")
        var_val = self.variance()
        std_val = self.standard_deviation()
        range_val = self.range_stat()
        iqr_val = self.iqr()
        
        print(f"\n--- Shape ---")
        skew_val = self.skewness()
        kurt_val = self.kurtosis()
        
        print(f"\n--- Percentiles ---")
        perc_val = self.percentiles()
        
        return {
            'mean': mean_val,
            'median': median_val,
            'mode': mode_val,
            'variance': var_val,
            'std': std_val,
            'range': range_val,
            'iqr': iqr_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'percentiles': perc_val
        }
    
    # ==================== VISUALIZATIONS ====================
    
    def visualize(self):
        """
        Create comprehensive visualizations of the data
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Descriptive Statistics Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0, 0].hist(self.data, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(np.mean(self.data), color='red', linestyle='--', label=f'Mean: {np.mean(self.data):.2f}')
        axes[0, 0].axvline(np.median(self.data), color='green', linestyle='--', label=f'Median: {np.median(self.data):.2f}')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Box Plot
        axes[0, 1].boxplot(self.data, vert=True)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Violin Plot
        parts = axes[0, 2].violinplot([self.data], vert=True, showmeans=True, showmedians=True)
        axes[0, 2].set_title('Violin Plot')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot (Quantile-Quantile)
        stats.probplot(self.data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Cumulative Distribution
        sorted_data = np.sort(self.data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cumulative, linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Density Plot
        axes[1, 2].hist(self.data, bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
        from scipy.stats import gaussian_kde
        density = gaussian_kde(self.data)
        xs = np.linspace(self.data.min(), self.data.max(), 200)
        axes[1, 2].plot(xs, density(xs), 'b-', linewidth=2, label='KDE')
        axes[1, 2].set_title('Density Plot')
        axes[1, 2].set_xlabel('Value')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/vercel/sandbox/descriptive_stats_visualization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'descriptive_stats_visualization.png'")
        plt.show()


# ==================== PRACTICAL EXAMPLES ====================

def example_1_basic():
    """Example 1: Analyzing exam scores"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Analyzing Student Exam Scores")
    print("="*60)
    
    # Sample data: exam scores out of 100
    scores = [78, 85, 92, 88, 76, 95, 89, 84, 91, 87, 
              82, 90, 86, 88, 93, 79, 85, 88, 92, 84]
    
    ds = DescriptiveStatistics(scores)
    summary = ds.summary()
    ds.visualize()
    
    print("\n📊 Key Insights:")
    print(f"- Average score: {summary['mean']:.1f}%")
    print(f"- Middle score: {summary['median']:.1f}%")
    print(f"- Score spread (std): ±{summary['std']:.1f} points")
    print(f"- Top 25% scored above: {summary['percentiles'][75]:.1f}%")


def example_2_skewed():
    """Example 2: Analyzing income data (skewed distribution)"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Analyzing Income Distribution (Skewed Data)")
    print("="*60)
    
    # Sample data: annual income in thousands
    np.random.seed(42)
    # Most people earn 30-80k, few earn much more (right-skewed)
    income = np.concatenate([
        np.random.normal(50, 15, 80),  # Majority
        np.random.normal(120, 30, 15),  # High earners
        np.random.normal(250, 50, 5)    # Very high earners
    ])
    
    ds = DescriptiveStatistics(income)
    summary = ds.summary()
    ds.visualize()
    
    print("\n📊 Key Insights:")
    print(f"- Mean income: ${summary['mean']:.1f}k (pulled up by high earners)")
    print(f"- Median income: ${summary['median']:.1f}k (typical person)")
    print(f"- Median is better measure here due to skewness: {summary['skewness']:.2f}")


def example_3_outliers():
    """Example 3: Effect of outliers"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Understanding Impact of Outliers")
    print("="*60)
    
    # Data with outliers
    data_with_outliers = [10, 12, 11, 13, 12, 14, 11, 13, 12, 100, 105]
    data_without_outliers = [10, 12, 11, 13, 12, 14, 11, 13, 12]
    
    print("\n--- WITH OUTLIERS ---")
    ds1 = DescriptiveStatistics(data_with_outliers)
    s1 = ds1.summary()
    
    print("\n--- WITHOUT OUTLIERS ---")
    ds2 = DescriptiveStatistics(data_without_outliers)
    s2 = ds2.summary()
    
    print("\n📊 Comparison:")
    print(f"Mean: {s1['mean']:.1f} (with) vs {s2['mean']:.1f} (without) - Changed by {abs(s1['mean']-s2['mean']):.1f}")
    print(f"Median: {s1['median']:.1f} (with) vs {s2['median']:.1f} (without) - Changed by {abs(s1['median']-s2['median']):.1f}")
    print(f"→ Median is more robust to outliers!")


# ==================== INTERACTIVE EXERCISES ====================

def exercise_1():
    """
    Exercise 1: Calculate statistics for your own data
    """
    print("\n" + "="*60)
    print("EXERCISE 1: Analyze Your Own Data")
    print("="*60)
    print("\nTask: Calculate descriptive statistics for this dataset:")
    print("Daily steps: [8500, 10200, 7800, 9500, 11000, 8200, 9800, 10500, 8900, 9200]")
    print("\nQuestions:")
    print("1. What is the mean number of steps?")
    print("2. What is the median?")
    print("3. What is the standard deviation?")
    print("4. On how many days did you walk more than the mean?")
    print("\n--- SOLUTION ---")
    
    steps = [8500, 10200, 7800, 9500, 11000, 8200, 9800, 10500, 8900, 9200]
    ds = DescriptiveStatistics(steps)
    summary = ds.summary()
    
    above_mean = sum(1 for s in steps if s > summary['mean'])
    print(f"\nAnswer 4: {above_mean} days had steps above the mean")


def exercise_2():
    """
    Exercise 2: Interpret real-world data
    """
    print("\n" + "="*60)
    print("EXERCISE 2: Interpret Statistical Measures")
    print("="*60)
    print("\nScenario: You're analyzing customer satisfaction scores (1-10)")
    
    satisfaction = [8, 9, 7, 8, 9, 10, 7, 8, 9, 8, 7, 9, 8, 10, 9, 8, 7, 8, 9, 8]
    ds = DescriptiveStatistics(satisfaction)
    summary = ds.summary()
    
    print("\nQuestions:")
    print("1. Is the distribution skewed? How do you know?")
    print("2. What percentage of customers rated 8 or above?")
    print("3. Is the data consistent or highly variable?")
    
    print("\n--- ANSWERS ---")
    print(f"1. Skewness = {summary['skewness']:.2f} → ", end="")
    if abs(summary['skewness']) < 0.5:
        print("Approximately symmetric")
    print(f"2. {sum(1 for s in satisfaction if s >= 8)/len(satisfaction)*100:.0f}% rated 8+")
    print(f"3. Std = {summary['std']:.2f} → Low variability, consistent ratings")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "🎓 " * 20)
    print("WELCOME TO MODULE 1: DESCRIPTIVE STATISTICS")
    print("🎓 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Understand measures of central tendency (mean, median, mode)")
    print("2. Learn measures of variability (variance, std, range, IQR)")
    print("3. Interpret measures of shape (skewness, kurtosis)")
    print("4. Visualize data distributions")
    print("5. Apply concepts to real-world scenarios")
    
    # Run examples
    example_1_basic()
    example_2_skewed()
    example_3_outliers()
    
    # Run exercises
    exercise_1()
    exercise_2()
    
    print("\n" + "="*60)
    print("✅ MODULE 1 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Mean is sensitive to outliers, median is robust")
    print("• Standard deviation measures typical deviation from mean")
    print("• Skewness tells us about distribution asymmetry")
    print("• Always visualize your data before analyzing")
    print("• Choose appropriate measures based on data characteristics")
    print("\n➡️  Next: Module 2 - Probability Theory")
