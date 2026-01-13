import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

######## LOAD AND PREPARE DATA ########

def load_data():
    """Load the cleaned capstone dataset"""
    df = pd.read_csv('./Cleaned_Data/clean_graduation_influencers.csv')
    print("=" * 80)
    print("DATA LOADED")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    return df

######## EXPLORATORY DATA ANALYSIS (EDA) ########

def descriptive_statistics(df):
    """Calculate and display descriptive statistics"""
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    # Select numeric columns
    numeric_cols = ['graduation_rate', 'teacher_hours', 'class_size', 'expenditure_pct_gdp']

    print("\nSummary Statistics:")
    print(df[numeric_cols].describe().round(2))

    # Statistics by country
    print("\n\nStatistics by Country:")
    print("-" * 80)
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        country_name = country_data['country_name'].iloc[0]
        print(f"\n{country_name} ({country}):")
        print(country_data[numeric_cols].describe().round(2))

    # Statistics by year
    print("\n\nStatistics by Year:")
    print("-" * 80)
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        print(f"\n{year}:")
        print(year_data[numeric_cols].describe().round(2))


def create_visualizations(df, output_dir='./Results'):
    """Create comprehensive visualizations"""
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    ######## Distribution of variables ########
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Variables', fontsize=16, fontweight='bold')

    variables = [
        ('graduation_rate', 'Graduation Rate (%)'),
        ('teacher_hours', 'Teacher Working Hours (annual)'),
        ('class_size', 'Average Class Size (students)'),
        ('expenditure_pct_gdp', 'Education Expenditure (% GDP)')
    ]

    for idx, (var, title) in enumerate(variables):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[var], bins=5, edgecolor='black', alpha=0.7)
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.axvline(df[var].mean(), color='red', linestyle='--', label=f'Mean: {df[var].mean():.2f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: distributions.png")
    plt.close()

    ######## Graduation rates by country over time ########
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        ax.plot(country_data['year'], country_data['graduation_rate'],
                marker='o', linewidth=2, markersize=8,
                label=country_data['country_name'].iloc[0])

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Graduation Rate (%)', fontsize=12)
    ax.set_title('Secondary Education Graduation Rates by Country (2017-2019)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/graduation_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved: graduation_trends.png")
    plt.close()

    ####### Correlation heatmap ########
    numeric_cols = ['graduation_rate', 'teacher_hours', 'class_size', 'expenditure_pct_gdp']
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Variables', fontsize=14, fontweight='bold', pad=20)

    # Rename labels for better readability
    labels = ['Graduation Rate', 'Teacher Hours', 'Class Size', 'Expenditure (% GDP)']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: correlation_heatmap.png")
    plt.close()

    ######## Scatterplots: Each Factor vs graduation rate ########
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Relationship Between Factors and Graduation Rates',
                 fontsize=16, fontweight='bold')

    factors = [
        ('teacher_hours', 'Teacher Working Hours (annual)'),
        ('class_size', 'Average Class Size (students)'),
        ('expenditure_pct_gdp', 'Education Expenditure (% GDP)')
    ]

    colors = {'DEU': 'blue', 'FRA': 'red', 'GBR': 'green'}

    for idx, (var, title) in enumerate(factors):
        ax = axes[idx]
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            ax.scatter(country_data[var], country_data['graduation_rate'],
                       s=100, alpha=0.6, label=country_data['country_name'].iloc[0],
                       color=colors[country])

        # Add trend line
        z = np.polyfit(df[var], df['graduation_rate'], 1)
        p = np.poly1d(z)
        ax.plot(df[var], p(df[var]), "k--", alpha=0.5, linewidth=2)

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Graduation Rate (%)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatterplots.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: scatterplots.png")
    plt.close()

    # 5. Country comparison - all variables
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison Across Countries', fontsize=16, fontweight='bold')

    variables_compare = [
        ('graduation_rate', 'Graduation Rate (%)', axes[0, 0]),
        ('teacher_hours', 'Teacher Hours (annual)', axes[0, 1]),
        ('class_size', 'Class Size (students)', axes[1, 0]),
        ('expenditure_pct_gdp', 'Expenditure (% GDP)', axes[1, 1])
    ]

    for var, title, ax in variables_compare:
        country_means = df.groupby('country_name')[var].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(country_means)), country_means.values,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xticks(range(len(country_means)))
        ax.set_xticklabels(country_means.index, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(f'Average {title} by Country')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/country_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: country_comparison.png")
    plt.close()

######## CORRELATION ANALYSIS ########

def correlation_analysis(df):
    """Perform detailed correlation analysis"""
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    numeric_cols = ['graduation_rate', 'teacher_hours', 'class_size', 'expenditure_pct_gdp']

    # Pearson correlation
    print("\nPearson Correlation Coefficients:")
    print("-" * 80)

    for col in ['teacher_hours', 'class_size', 'expenditure_pct_gdp']:
        corr, p_value = stats.pearsonr(df[col], df['graduation_rate'])

        # Determine significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Interpret strength
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.4:
            strength = "Moderate"
        elif abs_corr >= 0.2:
            strength = "Weak"
        else:
            strength = "Very weak"

        direction = "positive" if corr > 0 else "negative"

        print(f"\n{col} vs graduation_rate:")
        print(f"  Correlation: {corr:7.4f} {sig}")
        print(f"  P-value:     {p_value:7.4f}")
        print(f"  Interpretation: {strength} {direction} relationship")

    print("\n\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Full correlation matrix
    print("\n\nFull Correlation Matrix:")
    print("-" * 80)
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix.round(4))

######## MULTIPLE LINEAR REGRESSION ########

def multiple_regression_analysis(df):
    """Perform multiple linear regression analysis using sklearn and scipy"""
    print("MULTIPLE LINEAR REGRESSION ANALYSIS")
    print("=" * 80)

    # Prepare data
    X = df[['teacher_hours', 'class_size', 'expenditure_pct_gdp']].values
    y = df['graduation_rate'].values
    feature_names = ['teacher_hours', 'class_size', 'expenditure_pct_gdp']

    # Fit the model using sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)

    # Get predictions
    y_pred = model_sklearn.predict(X)
    residuals = y - y_pred

    # Calculate statistics manually
    n = len(y)
    k = X.shape[1]  # number of predictors

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Mean Squared Error and RMSE
    mse = ss_residual / (n - k - 1)
    rmse = np.sqrt(mse)

    # F-statistic
    ms_model = (ss_total - ss_residual) / k
    ms_residual = ss_residual / (n - k - 1)
    f_statistic = ms_model / ms_residual
    f_pvalue = 1 - stats.f.cdf(f_statistic, k, n - k - 1)

    # Standard errors and t-statistics for coefficients
    # Calculate standard errors
    X_with_intercept = np.column_stack([np.ones(n), X])
    xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    var_beta = mse * np.diag(xtx_inv)
    se_beta = np.sqrt(var_beta)

    # T-statistics
    all_coefs = np.concatenate([[model_sklearn.intercept_], model_sklearn.coef_])
    t_stats = all_coefs / se_beta

    # P-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

    # Confidence intervals (95%)
    t_crit = stats.t.ppf(0.975, n - k - 1)
    ci_lower = all_coefs - t_crit * se_beta
    ci_upper = all_coefs + t_crit * se_beta

    # Print results
    print("\n")
    print("REGRESSION RESULTS")
    print("=" * 80)

    print(f"\nModel Summary:")
    print(f"Dependent Variable: graduation_rate")
    print(f"Number of Observations: {n}")
    print(f"Number of Predictors: {k}")

    print(f"\n\nModel Fit:")
    print(f"R-squared:           {r_squared:.4f}")
    print(f"Adjusted R-squared:  {adj_r_squared:.4f}")
    print(f"F-statistic:         {f_statistic:.4f}")
    print(f"Prob (F-statistic):  {f_pvalue:.6f}")
    print(f"RMSE:                {rmse:.4f}")

    print(f"\n\nCoefficients:")
    print(f"{'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}")
    print("-" * 90)

    # Intercept
    print(
        f"{'Intercept':<25} {all_coefs[0]:>10.4f} {se_beta[0]:>10.4f} {t_stats[0]:>8.4f} {p_values[0]:>10.6f} {ci_lower[0]:>10.4f} {ci_upper[0]:>10.4f}")

    # Factors
    for i, name in enumerate(feature_names):
        idx = i + 1
        sig = ""
        if p_values[idx] < 0.001:
            sig = " ***"
        elif p_values[idx] < 0.01:
            sig = " **"
        elif p_values[idx] < 0.05:
            sig = " *"

        print(
            f"{name:<25} {all_coefs[idx]:>10.4f} {se_beta[idx]:>10.4f} {t_stats[idx]:>8.4f} {p_values[idx]:>10.6f} {ci_lower[idx]:>10.4f} {ci_upper[idx]:>10.4f}{sig}")

    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05")

    # Additional interpretations
    print("\n")
    print("MODEL INTERPRETATION")
    print("=" * 80)

    print(f"\nR-squared: {r_squared:.4f}")
    print(f"  → The model explains {r_squared * 100:.2f}% of variance in graduation rates")

    print(f"\nAdjusted R-squared: {adj_r_squared:.4f}")
    print(f"  → Adjusted for number of predictors: {adj_r_squared * 100:.2f}%")

    print(f"\nF-statistic: {f_statistic:.4f} (p-value: {f_pvalue:.6f})")
    if f_pvalue < 0.05:
        print("The model is statistically significant overall")
    else:
        print("The model is NOT statistically significant overall")

    print("\n\nCOEFFICIENT INTERPRETATION:")
    print("-" * 80)

    for i, var in enumerate(feature_names):
        coef = model_sklearn.coef_[i]
        p_val = p_values[i + 1]
        ci_low = ci_lower[i + 1]
        ci_up = ci_upper[i + 1]

        sig = ""
        if p_val < 0.001:
            sig = " ***"
        elif p_val < 0.01:
            sig = " **"
        elif p_val < 0.05:
            sig = " *"
        else:
            sig = " (ns)"

        print(f"\n{var}:")
        print(f"Coefficient: {coef:8.4f}{sig}")
        print(f"P-value:     {p_val:8.6f}")
        print(f"95% CI:      [{ci_low:7.4f}, {ci_up:7.4f}]")

        if p_val < 0.05:
            if 'teacher' in var.lower():
                print(f"For every 1 hour increase in teacher working hours,")
                print(f"graduation rate changes by {coef:.4f} percentage points")
            elif 'class' in var.lower():
                print(f"For every 1 student increase in class size,")
                print(f"graduation rate changes by {coef:.4f} percentage points")
            elif 'expenditure' in var.lower():
                print(f"For every 1% increase in education expenditure (as % GDP),")
                print(f"graduation rate changes by {coef:.4f} percentage points")
        else:
            print(f"NOT statistically significant (cannot conclude relationship)")

    # Create a model object to pass around
    model = {
        'sklearn_model': model_sklearn,
        'coef_': np.concatenate([[model_sklearn.intercept_], model_sklearn.coef_]),
        'params': {name: coef for name, coef in zip(['const'] + feature_names, all_coefs)},
        'pvalues': {name: pval for name, pval in zip(['const'] + feature_names, p_values)},
        'rsquared': r_squared,
        'rsquared_adj': adj_r_squared,
        'fvalue': f_statistic,
        'f_pvalue': f_pvalue,
        'residuals': residuals,
        'fittedvalues': y_pred,
        'conf_int': list(zip(ci_lower, ci_upper))
    }

    return model, df[['teacher_hours', 'class_size', 'expenditure_pct_gdp']], df['graduation_rate']

######## MODEL DIAGNOSTICS ########

def model_diagnostics(model, X, y, output_dir='./Results'):
    """Perform model diagnostic checks"""
    print("\n")
    print("MODEL DIAGNOSTICS")
    print("=" * 80)

    # Get residuals
    residuals = model['residuals']
    fitted = model['fittedvalues']

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regression Diagnostics', fontsize=16, fontweight='bold')

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)

    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Scale-Location
    standardized_resid = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1, 1].hist(residuals, bins=5, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: diagnostics.png")
    plt.close()

    # Test for normality (Shapiro-Wilk)
    print("\n\nNormality Test (Shapiro-Wilk):")
    print("-" * 80)
    stat, p_value = stats.shapiro(residuals)
    print(f"Statistic: {stat:.4f}")
    print(f"P-value:   {p_value:.4f}")
    if p_value > 0.05:
        print("Residuals appear normally distributed (p > 0.05)")
    else:
        print("Residuals may not be normally distributed (p < 0.05)")

    # Check for multicollinearity using correlation matrix
    print("\n\nMulticollinearity Check (Correlation Between Predictors):")
    print("-" * 80)
    X_df = pd.DataFrame(X, columns=['teacher_hours', 'class_size', 'expenditure_pct_gdp'])
    corr_matrix = X_df.corr()
    print(corr_matrix.round(4))
    print("\nInterpretation:")
    print("  |r| < 0.7: Low multicollinearity")
    print("  |r| 0.7-0.9: Moderate multicollinearity")
    print("  |r| > 0.9: High multicollinearity (problematic)")

######## HYPOTHESIS TESTING ########

def hypothesis_testing(df, model):
    """Formal hypothesis testing for each predictor"""
    print("\n")
    print("HYPOTHESIS TESTING")
    print("=" * 80)

    print("\n\nPRIMARY HYPOTHESIS:")
    print("-" * 80)
    print("H0: Class size, teacher working hours, and education expenditure")
    print("    do NOT have a significant impact on graduation rates")
    print("H1: At least one of these factors has a significant impact on")
    print("    graduation rates")

    # Overall F-test
    print(f"\n\nOverall Model F-test:")
    print(f"  F-statistic: {model['fvalue']:.4f}")
    print(f"  P-value:     {model['f_pvalue']:.6f}")

    if model['f_pvalue'] < 0.05:
        print(f"REJECT H0 (p < 0.05)")
        print(f"At least one predictor significantly impacts graduation rates")
    else:
        print(f"FAIL TO REJECT H0 (p ≥ 0.05)")
        print(f"Cannot conclude that predictors impact graduation rates")

    # Individual t-tests
    print("\n\nINDIVIDUAL PREDICTOR TESTS:")
    print("-" * 80)

    predictors = ['teacher_hours', 'class_size', 'expenditure_pct_gdp']
    predictor_names = {
        'teacher_hours': 'Teacher Working Hours',
        'class_size': 'Class Size',
        'expenditure_pct_gdp': 'Education Expenditure (% GDP)'
    }

    for var in predictors:
        print(f"\n{predictor_names[var]}:")
        print(f"H0: {var} has NO effect on graduation rates (β = 0)")
        print(f"H1: {var} has an effect on graduation rates (β ≠ 0)")

        coef = model['params'][var]
        p_val = model['pvalues'][var]

        # Calculate t-statistic from coefficient and p-value
        # (Already calculated in regression, but can show it)

        print(f"\nCoefficient (β): {coef:8.4f}")
        print(f"P-value:         {p_val:8.6f}")

        if p_val < 0.05:
            print(f"REJECT H0 (p < 0.05)")
            print(f"{predictor_names[var]} significantly impacts graduation rates")
        else:
            print(f"FAIL TO REJECT H0 (p ≥ 0.05)")
            print(f"Cannot conclude {predictor_names[var]} impacts graduation rates")

######## PREDICTIONS AND PRACTICAL SIGNIFICANCE ########

def predictions_and_practical_significance(model, df):
    """Make predictions and assess practical significance"""
    print("\n")
    print("PREDICTIONS AND PRACTICAL SIGNIFICANCE")
    print("=" * 80)

    # Get predictions
    predictions = model['fittedvalues']

    # Compare actual vs predicted
    results_df = pd.DataFrame({
        'Country': df['country_name'],
        'Year': df['year'],
        'Actual': df['graduation_rate'],
        'Predicted': predictions,
        'Residual': df['graduation_rate'] - predictions
    })

    print("\n\nACTUAL VS PREDICTED GRADUATION RATES:")
    print("-" * 80)
    print(results_df.to_string(index=False))

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(df['graduation_rate'], predictions))
    mae = np.mean(np.abs(df['graduation_rate'] - predictions))

    print(f"\n\nMODEL ACCURACY:")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f} percentage points")
    print(f"MAE (Mean Absolute Error):      {mae:.4f} percentage points")

    # Practical significance examples
    print("\n\nPRACTICAL SIGNIFICANCE EXAMPLES:")
    print("-" * 80)
    print("\nIf a country were to change one factor, holding others constant:")

    for var in ['teacher_hours', 'class_size', 'expenditure_pct_gdp']:
        coef = model['params'][var]
        p_val = model['pvalues'][var]

        if p_val < 0.05:  # Only show if significant
            if 'teacher' in var:
                change = 100  # 100 hours
                effect = coef * change
                print(f"\n• Increase teacher working hours by {change} hours:")
                print(f"  Expected change in graduation rate: {effect:+.2f} percentage points")
            elif 'class' in var:
                change = 5  # 5 students
                effect = coef * change
                print(f"\n• Increase average class size by {change} students:")
                print(f"  Expected change in graduation rate: {effect:+.2f} percentage points")
            elif 'expenditure' in var:
                change = 1  # 1% of GDP
                effect = coef * change
                print(f"\n• Increase education expenditure by {change}% of GDP:")
                print(f"  Expected change in graduation rate: {effect:+.2f} percentage points")

######## SUMMARY REPORT ########

def create_summary_report(df, model):
    """Create a comprehensive summary report"""
    print("\n")
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 80)

    print("\n\nRESEARCH QUESTION:")
    print("-" * 80)
    print("Do class size, hours that teachers work, and education expenditures")
    print("have an impact on graduation rates?")

    print("\n\nDATASET OVERVIEW:")
    print("-" * 80)
    print(f"Countries: Germany, France, United Kingdom (3 G20 countries)")
    print(f"Time Period: 2017-2019 (3 years)")
    print(f"Total Observations: {len(df)}")
    print(f"Variables: 4 (graduation rate + 3 factors)")

    print("\n\nKEY FINDINGS:")
    print("-" * 80)

    # Overall model
    print(f"\nOverall Model:")
    print(f"R² = {model['rsquared']:.4f} ({model['rsquared'] * 100:.2f}% of variance explained)")
    print(f"F-statistic = {model['fvalue']:.4f}, p = {model['f_pvalue']:.6f}")

    if model['f_pvalue'] < 0.05:
        print(f"Model is STATISTICALLY SIGNIFICANT")
    else:
        print(f"Model is NOT statistically significant")

    # Individual predictors
    print(f"\nIndividual Predictors:")
    for var in ['teacher_hours', 'class_size', 'expenditure_pct_gdp']:
        coef = model['params'][var]
        p_val = model['pvalues'][var]

        var_name = var.replace('_', ' ').title()
        print(f"\n{var_name}:")
        print(f"Coefficient: {coef:+.4f}")
        print(f"P-value: {p_val:.6f}")
        print(f"Significant: {'YES' if p_val < 0.05 else 'NO'}")

    print("\n\nCONCLUSIONS:")
    print("-" * 80)

    # Determine which hypothesis is supported
    sig_predictors = []
    for var in ['teacher_hours', 'class_size', 'expenditure_pct_gdp']:
        if model['pvalues'][var] < 0.05:
            sig_predictors.append(var.replace('_', ' '))

    if len(sig_predictors) > 0:
        print(f"\nThe analysis provides evidence that the following factors")
        print(f"significantly influence graduation rates:")
        for pred in sig_predictors:
            print(f"{pred.title()}")
    else:
        print(f"\nThe analysis does NOT provide strong statistical evidence that")
        print(f"these factors significantly influence graduation rates in this dataset.")


######## MAIN EXECUTION ########

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("  CAPSTONE STATISTICAL ANALYSIS")
    print("  Research Question: Impact of Teacher Factors on Graduation Rates")
    print("=" * 80)

    # Load data
    df = load_data()

    # Descriptive statistics
    descriptive_statistics(df)

    # Create visualizations
    create_visualizations(df)

    # Correlation analysis
    correlation_analysis(df)

    # Multiple regression
    model, X, y = multiple_regression_analysis(df)

    # Model diagnostics
    model_diagnostics(model, X, y)

    # Hypothesis testing
    hypothesis_testing(df, model)

    # Predictions and practical significance
    predictions_and_practical_significance(model, df)

    # Summary report
    create_summary_report(df, model)

    print("\n" + "*" * 80)
    print("STATISTICAL ANALYSIS COMPLETE!")
    print("All results and visualizations saved to ./Results")


if __name__ == "__main__":
    main()