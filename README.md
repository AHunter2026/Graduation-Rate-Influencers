## Project Overview

This capstone project investigates the relationship between 
certain factors and secondary education graduation rates 
across three G20 countries: Germany, France, and the United Kingdom. 
Using quantitative data from 2017-2019, the analysis examines whether 
class size, teacher working hours, and education expenditure significantly 
influence graduation outcomes.

### Research Question
Do class size, teacher working hours, and education expenditure have a 
significant impact on secondary education graduation rates in G20 countries?

### Key Finding
Education expenditure is the only statistically significant predictor of 
graduation rates among the three factors examined.

- β = +12.27 (p = 0.012)
- Each additional 1% of GDP spent on education is associated with a 12.27 
  percentage point increase in graduation rates
- Model R² = 83.8% (explains 83.8% of variance in graduation rates)
- Overall model: F = 8.60, p = 0.020 (statistically significant)

## Repository Structure

Root: Graduation Rate Influencers
Directory: Cleaned_Data
    - clean_class_size.csv
    - clean_expenditure.csv
    - clean_graduation_influencers.csv
    - clean_graduation_rates.csv
    - clean_teacher_hours.csv
Directory: Papers
    - Graduation Rate Influencers - Results.docx
    - Graduation Rate Influencers - The Proposal.docx
Directory: Raw_Data
    - OECD_Avg_Class_Size.csv
    - OECD_Hours_2017.XLSX
    - OECD_Hours_2018.XLSX
    - OECD_Hours_2019.XLSX
    - UNESCO_Graduation_Rates.csv
    - WB_Expenditures.csv
Directory: Results
    - correlation_heatmap.png
    - country_comparison.png
    - diagnostics.png
    - distributions.png
    - graduation_trends.png
    - scatterplots.png
    - statistical_results.txt
Directory: Scripts
    - Analysis.py
    - Cleaning_Merging.py
README.md (this file)
requirements.txt

## Methodology

### Data Sources

1. OECD Education at a Glance (2018, 2019, 2020 editions)
   - Teacher statutory working hours (annual)
   - Average class size (lower secondary/ISCED 2)
   - Source: https://www.oecd.org/education/education-at-a-glance/

2. World Bank Education Statistics
   - Government expenditure on education (% of GDP)
   - Source: https://data.worldbank.org/

3. UNESCO Institute for Statistics
   - Upper secondary graduation rates (%)
   - Source: http://data.uis.unesco.org/

### Countries & Time Period

- Countries: Germany (DEU), France (FRA), United Kingdom (GBR)
- Years: 2017, 2018, 2019
- Sample size: 9 observations (3 countries × 3 years)

### Variables

| Variable | Description | Source | Mean | Range |
|----------|-------------|--------|------|-------|
| Graduation Rate (dependent) | Upper secondary graduation rate (%) | UNESCO | 83.4% | 73.1% - 95.8% |
| Teacher Hours (predictor) | Annual statutory working hours | OECD | 1,505 hrs | 1,265 - 1,642 hrs |
| Class Size (predictor) | Average class size, lower secondary | OECD | 21.0 students | 19.0 - 24.3 students |
| Expenditure (predictor) | Education spending (% of GDP) | World Bank | 5.17% | 4.9% - 5.4% |

### Statistical Methods

1. Descriptive Statistics - Summary statistics for all variables
2. Exploratory Data Analysis - Distributions, trends, and patterns
3. Correlation Analysis - Pearson correlation coefficients
4. Multiple Linear Regression - Primary analytical method
      - Dependent variable: Graduation Rate
      - Predictors: Teacher Hours, Class Size, Expenditure
      - Model: `Graduation Rate = β₀ + β₁(Teacher Hours) + β₂(Class Size) + β₃(Expenditure) + ε`
5. Assumption Testing - Linearity, normality, homoscedasticity, multicollinearity
6. Hypothesis Testing - t-tests for individual predictors, F-test for overall model

## Key Results

### Regression Model Summary

| Predictor | Coefficient (β) | Std Error | t-statistic | p-value | Significant? |
|-----------|--------------|-----------|-------------|---------|-----------|
| Intercept | 75.23 | - | - | - | - |
| Teacher Hours | -0.042 | 0.036 | -1.18 | 0.289 |  No |
| Class Size | 9.10 | 7.01 | 1.30 | 0.249 |  No |
| Expenditure (% GDP) | 12.27 | 3.26 | 3.77 | 0.012 | YES |

Model Statistics:
- R² = 0.838 (83.8% of variance explained)
- Adjusted R² = 0.741
- F-statistic = 8.60
- F-test p-value = 0.020 (overall model is significant)

### Interpretation

Statistically Significant Finding:
- A 1 percentage point increase in education expenditure (as % of GDP) 
  is associated with a 12.27 percentage point increase in graduation rates, 
  holding other factors constant
- 95% Confidence Interval: [4.06, 20.47] - entirely positive

Practical Significance Example:
- If Germany increased education expenditure from 4.9% to 5.4% of GDP (matching France), 
  the model predicts graduation rates would increase from 79.5% to approximately 85.6%

Non-Significant Findings:
- Teacher working hours: No significant relationship (p = 0.289)
- Class size: No significant relationship (p = 0.249)
- High multicollinearity between teacher hours and class size (r = 0.996) makes it 
  difficult to isolate their individual effects

## Installation & Usage

### Prerequisites

Python 3.12+
pandas >= 2.1.0
numpy >= 1.24.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.11.0
scikit-learn >= 1.3.0
openpyxl >= 3.1.0

### Installation

# Clone the repository
git clone https://github.com/AHunter2026/Graduation-Rate-Influencers.git
cd Graduation-Rate-Influencers

# Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Analysis

Step 1: Data Cleaning & Merging
```bash
python Scripts/Cleaning_Merging.py
```
What it does:
- Extracts data from OECD Excel files (teacher hours, class size)
- Loads World Bank and UNESCO data
- Standardizes country names and year formats
- Handles missing values
- Merges all datasets on country and year
- Outputs cleaned datasets to `Graduation-Rate-Influencers/Cleaned_Data/` folder

Output: `clean_graduation_influencers.csv` (9 rows × 5 columns)

Step 2: Statistical Analysis
```bash
python Scripts/Analysis.py
```
What it does:
1. Loads cleaned data
2. Calculates descriptive statistics
3. Creates exploratory visualizations
4. Performs correlation analysis
5. Runs multiple linear regression
6. Tests model assumptions (normality, homoscedasticity, multicollinearity)
7. Conducts hypothesis tests
8. Generates all visualizations
9. Saves results to `Graduation-Rate-Influencers/Results/` folder

Outputs:
- 6 visualization files (PNG format) in `Results`
- `statistical_output.txt` with complete analysis results

## Visualizations

### 1. Correlation Heatmap
[Correlation Heatmap](Results/correlation_heatmap.png)

Shows relationships between all variables. 
Key finding: Strong positive correlation between expenditure and graduation rates (r = 0.859).

### 2. Expenditure vs. Graduation Rates
[Expenditure Scatter Plot](Results/scatterplots.png)

Clear positive linear relationship supporting the regression finding.

### 3. Country Comparison
[Country Comparison](Results/country_comparison.png)

France: Highest expenditure (5.4% GDP) → Highest graduation rates (87.5%)  
Germany: Lowest expenditure (4.9% GDP) → Lowest graduation rates (79.5%)

### 4. Regression Diagnostics
[Residual Plot](Results/diagnostics.png)
[Q-Q Plot](Results/diagnostics.png)

Model assumptions verified:
- Normality of residuals (Shapiro-Wilk p = 0.114)
- Homoscedasticity (random residual pattern)
- Multicollinearity detected (teacher hours & class size: r = 0.996)

## Policy Recommendations

### Recommendation 1: Prioritize Strategic Education Funding Increases

Evidence: 
Education expenditure is the only statistically significant 
predictor (p = 0.012), with both statistical and practical significance.

Specific Action:
- Countries spending below 5% of GDP on education should target increases toward 5.5-6%
- Each 0.5 percentage point increase could improve graduation rates by ~6 percentage points
- Track which spending categories (teacher salaries, infrastructure, materials) drive improvements

Example: 
Germany could potentially increase graduation rates from 79.5% to 85.6% by matching France's 
expenditure level (4.9% → 5.4% of GDP).

### Recommendation 2: Investigate Education Spending Composition

Evidence: 
The analysis found no significant relationship between teacher working hours or class size and 
graduation rates when controlling for expenditure. High multicollinearity (r = 0.996) suggests 
these factors move together.

Specific Action:
- Conduct detailed audits of education spending allocation
- Compare spending composition across high-performing vs. low-performing systems
- Focus on how money is spent rather than just total amount
- Research suggests quality of spending matters as much as quantity

Rationale: 
Simply mandating smaller class sizes or longer teacher hours may not improve outcomes. 
Strategic resource allocation is key.

## Limitations

1. Small Sample Size: Only 9 observations (3 countries × 3 years) limits statistical power and 
   generalizability
2. Limited Geographic Scope: Results apply only to Germany, France, and UK; may not generalize 
   to other education systems
3. Short Time Period: 3-year timeframe may not capture long-term policy effects
4. Correlation, Not Causation: Regression shows associations, not causal relationships
5. Multicollinearity: High correlation between teacher hours and class size (r = 0.996) makes it
   difficult to separate their effects
6. Aggregate Data: National-level data obscures within-country variation (regional, socioeconomic
   differences)
7. Limited Control Variables: Does not account for other factors affecting graduation rates 
   (student SES, prior achievement, curriculum quality)

## Future Research

1. Expand Sample: Include more countries and longer time periods for greater statistical power
2. Longitudinal Analysis: Track policy changes over time to assess causal effects
3. Spending Composition: Break down expenditure into specific categories (salaries, infrastructure, materials)
4. Multilevel Modeling: Account for within-country variation (regional, school-level differences)
5. Qualitative Research: Investigate how high-performing countries allocate resources effectively
6. Interaction Effects: Examine whether expenditure effects vary by country context or existing resource levels

## References

Antoniou, F., Alghamdi, M. H., & Kawai, K. (2024). 
The effect of school size and class size on school preparedness. 
*Frontiers in Psychology*, *15*, Article 1354072. https://doi.org/10.3389/fpsyg.2024.1354072

Boeskens, L., & Nusche, D. (2021). 
*Not enough hours in the day: Policies that shape teachers' use of time* 
(OECD Education Working Papers, No. 245). OECD Publishing. https://doi.org/10.1787/15990b42-en

OECD. (2023). 
*Education at a Glance 2023: OECD Indicators*. 
OECD Publishing. https://doi.org/10.1787/e13bef63-en

## Author

Ashley Hunter
Western Governors University  
Bachelor of Science in Data Analytics

## Acknowledgments

- Western Governors University for academic guidance and support
- OECD, World Bank, and UNESCO for providing publicly accessible education data

**Last Updated:** January 2026