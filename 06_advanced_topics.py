"""
Module 6: Advanced Topics - Statistics for Machine Learning
============================================================
Learn advanced statistical concepts essential for data science and ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ConfidenceIntervals:
    """
    Understanding and calculating confidence intervals
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("CONFIDENCE INTERVALS")
        print("="*60)
        print("\nA confidence interval gives a range of plausible values")
        print("for a population parameter.")
        print("\n95% CI: If we repeated sampling many times, 95% of")
        print("intervals would contain the true population parameter.")
        print("\nInterpretation: We are 95% confident the true value")
        print("lies within this interval.")
    
    @staticmethod
    def mean_ci(data, confidence=0.95):
        """
        Confidence interval for mean
        """
        print(f"\n--- {confidence*100:.0f}% Confidence Interval for Mean ---")
        
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)  # Standard error
        
        # T-distribution (for small samples)
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        
        print(f"Sample size: {n}")
        print(f"Sample mean: {mean:.4f}")
        print(f"Standard error: {se:.4f}")
        print(f"\n{confidence*100:.0f}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"\nInterpretation: We are {confidence*100:.0f}% confident the")
        print(f"true population mean is between {ci[0]:.4f} and {ci[1]:.4f}")
        
        return ci
    
    @staticmethod
    def proportion_ci(successes, n, confidence=0.95):
        """
        Confidence interval for proportion
        """
        print(f"\n--- {confidence*100:.0f}% Confidence Interval for Proportion ---")
        
        p_hat = successes / n
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        
        ci_lower = p_hat - z * se
        ci_upper = p_hat + z * se
        
        print(f"Sample size: {n}")
        print(f"Successes: {successes}")
        print(f"Sample proportion: {p_hat:.4f}")
        print(f"\n{confidence*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Or: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        
        return (ci_lower, ci_upper)
    
    @staticmethod
    def visualize_ci(data, confidence=0.95):
        """
        Visualize confidence interval
        """
        mean = np.mean(data)
        ci = stats.t.interval(confidence, len(data)-1, 
                             loc=mean, scale=stats.sem(data))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data
        ax.scatter(range(len(data)), data, alpha=0.5, s=50, label='Data points')
        
        # Plot mean
        ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
        
        # Plot CI
        ax.axhspan(ci[0], ci[1], alpha=0.2, color='green', 
                  label=f'{confidence*100:.0f}% CI')
        ax.axhline(ci[0], color='green', linestyle=':', linewidth=2)
        ax.axhline(ci[1], color='green', linestyle=':', linewidth=2)
        
        ax.set_xlabel('Observation')
        ax.set_ylabel('Value')
        ax.set_title(f'Data with {confidence*100:.0f}% Confidence Interval')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class ANOVA:
    """
    Analysis of Variance - Comparing multiple groups
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("ANOVA (Analysis of Variance)")
        print("="*60)
        print("\nUse: Compare means of 3+ groups")
        print("H₀: All group means are equal")
        print("H₁: At least one group mean is different")
        print("\nWhy not multiple t-tests?")
        print("→ Multiple comparisons increase Type I error rate")
        print("→ ANOVA controls overall error rate")
    
    @staticmethod
    def one_way_anova(*groups, alpha=0.05):
        """
        One-way ANOVA
        """
        print("\n--- One-Way ANOVA ---")
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate group statistics
        print(f"\nNumber of groups: {len(groups)}")
        for i, group in enumerate(groups, 1):
            print(f"Group {i}: n={len(group)}, mean={np.mean(group):.2f}, std={np.std(group, ddof=1):.2f}")
        
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < alpha:
            print(f"\n✓ REJECT H₀ (p={p_value:.4f} < α={alpha})")
            print("Conclusion: At least one group mean is significantly different")
            print("→ Perform post-hoc tests to identify which groups differ")
        else:
            print(f"\n✗ FAIL TO REJECT H₀ (p={p_value:.4f} ≥ α={alpha})")
            print("Conclusion: No significant difference between group means")
        
        return f_stat, p_value
    
    @staticmethod
    def visualize_anova(*groups, group_names=None):
        """
        Visualize ANOVA
        """
        if group_names is None:
            group_names = [f'Group {i+1}' for i in range(len(groups))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        ax1.boxplot(groups, labels=group_names)
        ax1.set_ylabel('Value')
        ax1.set_title('Group Comparison (Box Plot)')
        ax1.grid(True, alpha=0.3)
        
        # Violin plot
        positions = range(1, len(groups) + 1)
        parts = ax2.violinplot(groups, positions=positions, showmeans=True, showmedians=True)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(group_names)
        ax2.set_ylabel('Value')
        ax2.set_title('Group Comparison (Violin Plot)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class BootstrapResampling:
    """
    Bootstrap: Resampling method for estimating uncertainty
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("BOOTSTRAP RESAMPLING")
        print("="*60)
        print("\nBootstrap: Resampling with replacement to estimate")
        print("sampling distribution and uncertainty.")
        print("\nProcess:")
        print("1. Resample data with replacement (same size)")
        print("2. Calculate statistic (mean, median, etc.)")
        print("3. Repeat many times (e.g., 10,000)")
        print("4. Use distribution of statistics for inference")
        print("\nAdvantages:")
        print("• No assumptions about population distribution")
        print("• Works for any statistic")
        print("• Provides confidence intervals")
    
    @staticmethod
    def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, confidence=0.95):
        """
        Bootstrap confidence interval
        """
        print(f"\n--- Bootstrap {confidence*100:.0f}% CI ---")
        
        # Original statistic
        original_stat = statistic(data)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate CI
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        print(f"Original statistic: {original_stat:.4f}")
        print(f"Bootstrap iterations: {n_bootstrap}")
        print(f"Bootstrap mean: {np.mean(bootstrap_stats):.4f}")
        print(f"Bootstrap std: {np.std(bootstrap_stats):.4f}")
        print(f"\n{confidence*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(bootstrap_stats, bins=50, density=True, alpha=0.7, 
               edgecolor='black', color='skyblue')
        ax.axvline(original_stat, color='red', linestyle='--', 
                  linewidth=2, label=f'Original = {original_stat:.2f}')
        ax.axvline(ci_lower, color='green', linestyle=':', 
                  linewidth=2, label=f'CI [{ci_lower:.2f}, {ci_upper:.2f}]')
        ax.axvline(ci_upper, color='green', linestyle=':', linewidth=2)
        ax.set_xlabel('Statistic Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap Distribution ({n_bootstrap} iterations)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ci_lower, ci_upper, fig


class BiasVarianceTradeoff:
    """
    Understanding bias-variance tradeoff in ML
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("BIAS-VARIANCE TRADEOFF")
        print("="*60)
        print("\nTotal Error = Bias² + Variance + Irreducible Error")
        print("\nBIAS:")
        print("• Error from wrong assumptions")
        print("• Underfitting: Model too simple")
        print("• High bias → systematic errors")
        print("\nVARIANCE:")
        print("• Error from sensitivity to training data")
        print("• Overfitting: Model too complex")
        print("• High variance → inconsistent predictions")
        print("\nTRADEOFF:")
        print("• Simple models: High bias, low variance")
        print("• Complex models: Low bias, high variance")
        print("• Goal: Find optimal balance")
    
    @staticmethod
    def visualize_tradeoff():
        """
        Visualize bias-variance tradeoff
        """
        complexity = np.linspace(0, 10, 100)
        
        # Simulate bias and variance
        bias = 10 / (1 + complexity)
        variance = complexity / 2
        total_error = bias**2 + variance + 1  # +1 for irreducible error
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(complexity, bias**2, linewidth=2, label='Bias²', color='blue')
        ax.plot(complexity, variance, linewidth=2, label='Variance', color='red')
        ax.plot(complexity, total_error, linewidth=3, label='Total Error', 
               color='black', linestyle='--')
        
        # Mark optimal point
        optimal_idx = np.argmin(total_error)
        ax.axvline(complexity[optimal_idx], color='green', linestyle=':', 
                  linewidth=2, label='Optimal Complexity')
        
        ax.set_xlabel('Model Complexity')
        ax.set_ylabel('Error')
        ax.set_title('Bias-Variance Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Underfitting\n(High Bias)', xy=(2, 8), fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.annotate('Overfitting\n(High Variance)', xy=(8, 6), fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        return fig


class CrossValidation:
    """
    Cross-validation for model evaluation
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)
        print("\nPurpose: Assess model performance on unseen data")
        print("\nK-Fold Cross-Validation:")
        print("1. Split data into K folds")
        print("2. Train on K-1 folds, test on remaining fold")
        print("3. Repeat K times (each fold used as test once)")
        print("4. Average performance across all folds")
        print("\nAdvantages:")
        print("• Uses all data for training and testing")
        print("• Reduces variance in performance estimate")
        print("• Detects overfitting")
    
    @staticmethod
    def demonstrate_cv():
        """
        Demonstrate cross-validation
        """
        print("\n--- Cross-Validation Demo ---")
        
        # Generate data
        X, y = make_classification(n_samples=200, n_features=20, 
                                   n_informative=15, random_state=42)
        
        # Create model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        print(f"\n5-Fold Cross-Validation Results:")
        for i, score in enumerate(cv_scores, 1):
            print(f"Fold {i}: {score:.4f}")
        
        print(f"\nMean accuracy: {cv_scores.mean():.4f}")
        print(f"Std deviation: {cv_scores.std():.4f}")
        print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
              f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, 6), cv_scores, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axhline(cv_scores.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean = {cv_scores.mean():.3f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('5-Fold Cross-Validation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class ConfusionMatrixAnalysis:
    """
    Understanding confusion matrix and classification metrics
    """
    
    @staticmethod
    def explain():
        print("\n" + "="*60)
        print("CONFUSION MATRIX & CLASSIFICATION METRICS")
        print("="*60)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("              Positive  Negative")
        print("Actual Pos      TP        FN")
        print("       Neg      FP        TN")
        print("\nMetrics:")
        print("• Accuracy = (TP + TN) / Total")
        print("• Precision = TP / (TP + FP) - Of predicted positive, how many correct?")
        print("• Recall = TP / (TP + FN) - Of actual positive, how many found?")
        print("• F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
        print("\nWhen to use:")
        print("• Precision: When false positives are costly")
        print("• Recall: When false negatives are costly")
        print("• F1: Balance between precision and recall")
    
    @staticmethod
    def analyze_confusion_matrix(y_true, y_pred, labels=None):
        """
        Analyze confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n--- Confusion Matrix ---")
        print(cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n--- Metrics ---")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels or ['Negative', 'Positive'],
                   yticklabels=labels or ['Negative', 'Positive'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        
        return fig


# ==================== PRACTICAL EXAMPLES ====================

def example_1_confidence_intervals():
    """Example 1: Confidence intervals"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Confidence Intervals")
    print("="*60)
    
    ConfidenceIntervals.explain()
    
    # Example data: customer satisfaction scores
    np.random.seed(42)
    scores = np.random.normal(7.5, 1.5, 50)
    
    print("\n--- Customer Satisfaction Scores ---")
    ConfidenceIntervals.mean_ci(scores, confidence=0.95)
    
    fig = ConfidenceIntervals.visualize_ci(scores)
    plt.savefig('/vercel/sandbox/confidence_interval.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()
    
    # Proportion example
    print("\n--- Survey: Will customers recommend? ---")
    ConfidenceIntervals.proportion_ci(successes=75, n=100, confidence=0.95)


def example_2_anova():
    """Example 2: ANOVA"""
    print("\n" + "="*60)
    print("EXAMPLE 2: ANOVA - Comparing Multiple Groups")
    print("="*60)
    
    ANOVA.explain()
    
    print("\n--- Example: Comparing 3 teaching methods ---")
    
    # Generate data
    np.random.seed(42)
    method_a = np.random.normal(75, 10, 30)
    method_b = np.random.normal(80, 10, 30)
    method_c = np.random.normal(78, 10, 30)
    
    ANOVA.one_way_anova(method_a, method_b, method_c)
    
    fig = ANOVA.visualize_anova(method_a, method_b, method_c,
                                group_names=['Method A', 'Method B', 'Method C'])
    plt.savefig('/vercel/sandbox/anova_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_3_bootstrap():
    """Example 3: Bootstrap"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Bootstrap Resampling")
    print("="*60)
    
    BootstrapResampling.explain()
    
    # Example data
    np.random.seed(42)
    data = np.random.exponential(scale=10, size=50)
    
    print("\n--- Bootstrap CI for Median ---")
    ci_lower, ci_upper, fig = BootstrapResampling.bootstrap_ci(
        data, statistic=np.median, n_bootstrap=10000
    )
    
    plt.savefig('/vercel/sandbox/bootstrap_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_4_bias_variance():
    """Example 4: Bias-variance tradeoff"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Bias-Variance Tradeoff")
    print("="*60)
    
    BiasVarianceTradeoff.explain()
    
    fig = BiasVarianceTradeoff.visualize_tradeoff()
    plt.savefig('/vercel/sandbox/bias_variance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_5_cross_validation():
    """Example 5: Cross-validation"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Cross-Validation")
    print("="*60)
    
    CrossValidation.explain()
    
    fig = CrossValidation.demonstrate_cv()
    plt.savefig('/vercel/sandbox/cross_validation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_6_confusion_matrix():
    """Example 6: Confusion matrix"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Confusion Matrix Analysis")
    print("="*60)
    
    ConfusionMatrixAnalysis.explain()
    
    # Example predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(100, 20, replace=False)
    y_pred[errors] = 1 - y_pred[errors]
    
    fig = ConfusionMatrixAnalysis.analyze_confusion_matrix(y_true, y_pred)
    plt.savefig('/vercel/sandbox/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("WELCOME TO MODULE 6: ADVANCED TOPICS")
    print("🚀 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Calculate and interpret confidence intervals")
    print("2. Compare multiple groups with ANOVA")
    print("3. Use bootstrap for uncertainty estimation")
    print("4. Understand bias-variance tradeoff")
    print("5. Apply cross-validation")
    print("6. Interpret confusion matrices and classification metrics")
    
    # Run examples
    example_1_confidence_intervals()
    example_2_anova()
    example_3_bootstrap()
    example_4_bias_variance()
    example_5_cross_validation()
    example_6_confusion_matrix()
    
    print("\n" + "="*60)
    print("✅ MODULE 6 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Confidence intervals quantify uncertainty")
    print("• ANOVA compares 3+ groups simultaneously")
    print("• Bootstrap works without distribution assumptions")
    print("• Balance bias and variance for optimal models")
    print("• Cross-validation prevents overfitting")
    print("• Choose metrics based on problem context")
    print("\n🎉 CONGRATULATIONS! You've completed all modules!")
    print("You now have a solid foundation in statistics for data science!")
