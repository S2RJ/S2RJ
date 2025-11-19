"""
Module 5: Regression Analysis - Modeling Relationships
=======================================================
Learn how to model and predict relationships between variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SimpleLinearRegression:
    """
    Simple Linear Regression: Y = β₀ + β₁X + ε
    One predictor variable
    """
    
    def __init__(self, X, y):
        self.X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        self.y = np.array(y)
        self.n = len(y)
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        
    def explain(self):
        """Explain the regression equation"""
        print("\n" + "="*60)
        print("SIMPLE LINEAR REGRESSION")
        print("="*60)
        
        beta_0 = self.model.intercept_
        beta_1 = self.model.coef_[0]
        
        print(f"\nRegression Equation: Y = {beta_0:.4f} + {beta_1:.4f}X")
        print(f"\nInterpretation:")
        print(f"• Intercept (β₀): {beta_0:.4f}")
        print(f"  → Predicted Y when X = 0")
        print(f"• Slope (β₁): {beta_1:.4f}")
        print(f"  → For each 1-unit increase in X, Y changes by {beta_1:.4f}")
        
        return beta_0, beta_1
    
    def evaluate(self):
        """Evaluate model performance"""
        y_pred = self.model.predict(self.X)
        
        # R-squared
        r2 = r2_score(self.y, y_pred)
        
        # Adjusted R-squared
        k = 1  # number of predictors
        adj_r2 = 1 - (1 - r2) * (self.n - 1) / (self.n - k - 1)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        # MAE
        mae = mean_absolute_error(self.y, y_pred)
        
        # Correlation
        correlation = np.corrcoef(self.X.flatten(), self.y)[0, 1]
        
        print(f"\n--- Model Performance ---")
        print(f"R² (R-squared): {r2:.4f}")
        print(f"  → {r2*100:.2f}% of variance in Y explained by X")
        print(f"Adjusted R²: {adj_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"  → Average prediction error")
        print(f"MAE: {mae:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        return {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae}
    
    def hypothesis_test(self):
        """Test if slope is significantly different from zero"""
        print(f"\n--- Hypothesis Test for Slope ---")
        print(f"H₀: β₁ = 0 (no relationship)")
        print(f"H₁: β₁ ≠ 0 (significant relationship)")
        
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        
        # Standard error of slope
        mse = np.sum(residuals**2) / (self.n - 2)
        x_mean = np.mean(self.X)
        se_slope = np.sqrt(mse / np.sum((self.X - x_mean)**2))
        
        # T-statistic
        t_stat = self.model.coef_[0] / se_slope
        
        # P-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.n - 2))
        
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"✓ REJECT H₀: Slope is significantly different from zero")
        else:
            print(f"✗ FAIL TO REJECT H₀: Slope not significantly different from zero")
        
        return t_stat, p_value
    
    def visualize(self, xlabel='X', ylabel='Y', title='Linear Regression'):
        """Visualize regression line and residuals"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        
        # 1. Scatter plot with regression line
        axes[0].scatter(self.X, self.y, alpha=0.6, s=50, edgecolors='black')
        axes[0].plot(self.X, y_pred, 'r-', linewidth=2, label='Regression Line')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (Normality of Residuals)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def predict(self, X_new):
        """Make predictions"""
        X_new = np.array(X_new).reshape(-1, 1)
        predictions = self.model.predict(X_new)
        
        print(f"\n--- Predictions ---")
        for x, pred in zip(X_new.flatten(), predictions):
            print(f"X = {x:.2f} → Predicted Y = {pred:.2f}")
        
        return predictions


class MultipleLinearRegression:
    """
    Multiple Linear Regression: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
    Multiple predictor variables
    """
    
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = len(y)
        self.k = X.shape[1]
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
    
    def explain(self):
        """Explain the regression equation"""
        print("\n" + "="*60)
        print("MULTIPLE LINEAR REGRESSION")
        print("="*60)
        
        beta_0 = self.model.intercept_
        betas = self.model.coef_
        
        equation = f"Y = {beta_0:.4f}"
        for i, beta in enumerate(betas, 1):
            equation += f" + {beta:.4f}X{i}"
        
        print(f"\nRegression Equation: {equation}")
        print(f"\nCoefficients:")
        print(f"• Intercept: {beta_0:.4f}")
        for i, beta in enumerate(betas, 1):
            print(f"• β{i}: {beta:.4f}")
            print(f"  → Holding other variables constant, 1-unit increase in X{i}")
            print(f"    changes Y by {beta:.4f}")
        
        return beta_0, betas
    
    def evaluate(self):
        """Evaluate model performance"""
        y_pred = self.model.predict(self.X)
        
        r2 = r2_score(self.y, y_pred)
        adj_r2 = 1 - (1 - r2) * (self.n - 1) / (self.n - self.k - 1)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        mae = mean_absolute_error(self.y, y_pred)
        
        print(f"\n--- Model Performance ---")
        print(f"R²: {r2:.4f} ({r2*100:.2f}% variance explained)")
        print(f"Adjusted R²: {adj_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae}
    
    def feature_importance(self, feature_names=None):
        """Show feature importance"""
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(self.k)]
        
        print(f"\n--- Feature Importance (Absolute Coefficients) ---")
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(importance.to_string(index=False))
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Coefficient'], 
               color=['green' if c > 0 else 'red' for c in importance['Coefficient']],
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Importance in Multiple Regression')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        return fig


class PolynomialRegressionAnalysis:
    """
    Polynomial Regression: Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε
    For non-linear relationships
    """
    
    def __init__(self, X, y, degree=2):
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y)
        self.degree = degree
        
        # Create polynomial features
        self.poly = PolynomialFeatures(degree=degree)
        X_poly = self.poly.fit_transform(self.X)
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_poly, self.y)
    
    def explain(self):
        """Explain polynomial regression"""
        print("\n" + "="*60)
        print(f"POLYNOMIAL REGRESSION (Degree {self.degree})")
        print("="*60)
        
        print(f"\nFits a polynomial curve to capture non-linear relationships")
        print(f"Degree {self.degree}: Y = β₀ + β₁X + β₂X² + ... + β{self.degree}X^{self.degree}")
    
    def evaluate(self):
        """Evaluate model"""
        X_poly = self.poly.transform(self.X)
        y_pred = self.model.predict(X_poly)
        
        r2 = r2_score(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        print(f"\n--- Model Performance ---")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return {'r2': r2, 'rmse': rmse}
    
    def visualize(self):
        """Visualize polynomial fit"""
        X_plot = np.linspace(self.X.min(), self.X.max(), 300).reshape(-1, 1)
        X_plot_poly = self.poly.transform(X_plot)
        y_plot = self.model.predict(X_plot_poly)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.X, self.y, alpha=0.6, s=50, edgecolors='black', label='Data')
        ax.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Polynomial (degree {self.degree})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Polynomial Regression (Degree {self.degree})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


# ==================== PRACTICAL EXAMPLES ====================

def example_1_simple_regression():
    """Example 1: Simple linear regression"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Linear Regression - Advertising & Sales")
    print("="*60)
    
    print("\nScenario: Predicting sales from advertising spend")
    
    # Generate data
    np.random.seed(42)
    advertising = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    sales = 50 + 2.5 * advertising + np.random.normal(0, 5, 10)
    
    print(f"\nAdvertising (thousands): {advertising}")
    print(f"Sales (thousands): {sales.round(2)}")
    
    # Fit model
    slr = SimpleLinearRegression(advertising, sales)
    slr.explain()
    slr.evaluate()
    slr.hypothesis_test()
    
    # Predict
    print("\n--- Making Predictions ---")
    slr.predict([60, 70, 80])
    
    # Visualize
    fig = slr.visualize(xlabel='Advertising Spend ($1000s)', 
                       ylabel='Sales ($1000s)',
                       title='Advertising vs Sales')
    plt.savefig('/vercel/sandbox/simple_regression.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_2_multiple_regression():
    """Example 2: Multiple linear regression"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Regression - House Price Prediction")
    print("="*60)
    
    print("\nScenario: Predicting house prices from multiple features")
    
    # Generate data
    np.random.seed(42)
    n = 100
    size = np.random.uniform(1000, 3000, n)  # Square feet
    bedrooms = np.random.randint(2, 6, n)
    age = np.random.uniform(0, 50, n)  # Years
    
    # Price = 50000 + 100*size + 20000*bedrooms - 1000*age + noise
    price = (50000 + 100*size + 20000*bedrooms - 1000*age + 
             np.random.normal(0, 20000, n))
    
    X = np.column_stack([size, bedrooms, age])
    
    # Fit model
    mlr = MultipleLinearRegression(X, price)
    mlr.explain()
    mlr.evaluate()
    
    # Feature importance
    fig = mlr.feature_importance(['Size (sqft)', 'Bedrooms', 'Age (years)'])
    plt.savefig('/vercel/sandbox/multiple_regression.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    plt.show()


def example_3_polynomial():
    """Example 3: Polynomial regression"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Polynomial Regression - Non-linear Relationship")
    print("="*60)
    
    print("\nScenario: Temperature vs Ice Cream Sales (non-linear)")
    
    # Generate non-linear data
    np.random.seed(42)
    temperature = np.linspace(0, 40, 50)
    sales = 10 + 2*temperature - 0.03*temperature**2 + np.random.normal(0, 3, 50)
    
    # Compare different degrees
    degrees = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, degree in enumerate(degrees):
        poly_reg = PolynomialRegressionAnalysis(temperature, sales, degree=degree)
        poly_reg.explain()
        metrics = poly_reg.evaluate()
        
        # Plot
        X_plot = np.linspace(0, 40, 300).reshape(-1, 1)
        X_plot_poly = poly_reg.poly.transform(X_plot)
        y_plot = poly_reg.model.predict(X_plot_poly)
        
        axes[idx].scatter(temperature, sales, alpha=0.6, s=30, edgecolors='black')
        axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2)
        axes[idx].set_xlabel('Temperature (°C)')
        axes[idx].set_ylabel('Ice Cream Sales')
        axes[idx].set_title(f'Degree {degree} (R²={metrics["r2"]:.3f})')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/vercel/sandbox/polynomial_regression.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison saved")
    plt.show()


def example_4_assumptions():
    """Example 4: Checking regression assumptions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Checking Regression Assumptions")
    print("="*60)
    
    print("\nLinear regression assumptions:")
    print("1. Linearity: Relationship between X and Y is linear")
    print("2. Independence: Observations are independent")
    print("3. Homoscedasticity: Constant variance of residuals")
    print("4. Normality: Residuals are normally distributed")
    print("5. No multicollinearity (for multiple regression)")
    
    # Generate data
    np.random.seed(42)
    X = np.random.uniform(0, 10, 100)
    y = 2 + 3*X + np.random.normal(0, 2, 100)
    
    slr = SimpleLinearRegression(X, y)
    
    print("\n--- Diagnostic Plots ---")
    fig = slr.visualize(title='Regression Diagnostics')
    plt.savefig('/vercel/sandbox/regression_diagnostics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Diagnostics saved")
    
    print("\nInterpretation:")
    print("• Residual plot: Should show random scatter (no pattern)")
    print("• Q-Q plot: Points should follow diagonal line (normality)")
    plt.show()


def example_5_regularization():
    """Example 5: Ridge and Lasso regression"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Regularization - Ridge & Lasso")
    print("="*60)
    
    print("\nRegularization prevents overfitting by penalizing large coefficients")
    print("• Ridge (L2): Shrinks coefficients")
    print("• Lasso (L1): Can set coefficients to exactly zero (feature selection)")
    
    # Generate data with many features
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    # Only first 5 features are truly important
    true_coef = np.zeros(p)
    true_coef[:5] = [3, -2, 1.5, -1, 0.5]
    y = X @ true_coef + np.random.randn(n) * 0.5
    
    # Fit models
    lr = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.1).fit(X, y)
    
    # Compare coefficients
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(p)
    width = 0.25
    
    ax.bar(x_pos - width, lr.coef_, width, label='Linear Regression', alpha=0.7)
    ax.bar(x_pos, ridge.coef_, width, label='Ridge', alpha=0.7)
    ax.bar(x_pos + width, lasso.coef_, width, label='Lasso', alpha=0.7)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Comparison: Linear vs Ridge vs Lasso')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('/vercel/sandbox/regularization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")
    
    print(f"\nNumber of non-zero coefficients:")
    print(f"• Linear Regression: {np.sum(np.abs(lr.coef_) > 0.01)}")
    print(f"• Ridge: {np.sum(np.abs(ridge.coef_) > 0.01)}")
    print(f"• Lasso: {np.sum(np.abs(lasso.coef_) > 0.01)} (feature selection!)")
    
    plt.show()


# ==================== INTERACTIVE EXERCISES ====================

def exercise_1():
    """Exercise: Interpret regression output"""
    print("\n" + "="*60)
    print("EXERCISE 1: Interpret Regression Results")
    print("="*60)
    
    print("\nScenario: Study hours vs Exam score")
    print("Regression equation: Score = 40 + 5*Hours")
    print("R² = 0.75, p-value < 0.001")
    
    print("\nQuestions:")
    print("1. What does the intercept mean?")
    print("2. What does the slope mean?")
    print("3. What does R² = 0.75 mean?")
    print("4. Is the relationship significant?")
    
    print("\n--- ANSWERS ---")
    print("1. Intercept (40): Expected score with 0 hours of study")
    print("2. Slope (5): Each additional hour increases score by 5 points")
    print("3. R² = 0.75: 75% of variance in scores explained by study hours")
    print("4. Yes, p < 0.001 means highly significant relationship")


def exercise_2():
    """Exercise: Build your own regression"""
    print("\n" + "="*60)
    print("EXERCISE 2: Build a Regression Model")
    print("="*60)
    
    print("\nTask: Predict salary from years of experience")
    
    # Data
    experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    salary = np.array([35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    
    print(f"\nExperience (years): {experience}")
    print(f"Salary ($1000s): {salary}")
    
    print("\n--- SOLUTION ---")
    slr = SimpleLinearRegression(experience, salary)
    slr.explain()
    slr.evaluate()
    
    print("\nPredict salary for 12 years experience:")
    slr.predict([12])


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "📈 " * 20)
    print("WELCOME TO MODULE 5: REGRESSION ANALYSIS")
    print("📈 " * 20)
    
    print("\n📚 Learning Objectives:")
    print("1. Understand simple and multiple linear regression")
    print("2. Interpret regression coefficients and R²")
    print("3. Test significance of relationships")
    print("4. Handle non-linear relationships with polynomial regression")
    print("5. Check regression assumptions")
    print("6. Apply regularization techniques")
    
    # Run examples
    example_1_simple_regression()
    example_2_multiple_regression()
    example_3_polynomial()
    example_4_assumptions()
    example_5_regularization()
    
    # Run exercises
    exercise_1()
    exercise_2()
    
    print("\n" + "="*60)
    print("✅ MODULE 5 COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Regression models relationships between variables")
    print("• R² measures proportion of variance explained")
    print("• Check assumptions: linearity, normality, homoscedasticity")
    print("• Multiple regression: control for confounding variables")
    print("• Polynomial regression: capture non-linear patterns")
    print("• Regularization: prevent overfitting")
    print("\n➡️  Next: Module 6 - Advanced Topics & Machine Learning Statistics")
