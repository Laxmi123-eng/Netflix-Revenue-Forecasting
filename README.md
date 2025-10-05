#  Netflix Quarterly Revenue Forecasting (Time Series Project)

##  Overview
I developed a **time-series forecasting model** to predict Netflix’s quarterly global revenue using both historical data and regional business drivers such as memberships and ARPU.  
I applied **ARIMA** and **SARIMAX (ARIMA with Exogenous Variables)** models to capture trend, quarterly seasonality, and external business dynamics.  
The model forecasts Netflix’s future revenue trajectory and quantifies prediction uncertainty through confidence intervals.

---

##  Objectives
- Analyze quarterly Netflix revenue trends from 2019–2024  
- Build a baseline **ARIMA (1,1,1)** model for pure trend forecasting  
- Improve accuracy with **SARIMAX (1,1,1)** by adding membership and ARPU features  
- Forecast **8 future quarters (2024–2026)** with uncertainty bands  
- Visualize and compare model performance  

---

##  Methodology
###  Data Preparation
- Cleaned and formatted quarterly Netflix revenue and membership data  
- Parsed `Date`, converted all numeric columns, handled `$` and commas  
- Created lag and growth-rate features for additional signal  
- Split dataset chronologically (80 % train / 20 % test)

###  Modeling
- **ARIMA (1,1,1)** baseline for trend and noise components  
- **SARIMAX (1,1,1)** with exogenous regressors (Members + ARPU)  
- Evaluated forecast accuracy with **RMSE**

###  Evaluation
| Model             | RMSE (USD)     | Notes                              |
|-------------------|----------------|------------------------------------|
| ARIMA (1,1,1)     | $661,203,989   | Baseline (trend only)              |
| SARIMAX (1,1,1)   | $146,971,322   | Members + ARPU exogenous features  |

**Improvement:** SARIMAX reduced error by **≈77.8%** compared to ARIMA.

###  Extended Forecast
- Forecasted 8 future quarters (2024–2026)  
- Extrapolated membership + ARPU growth using recent trends + mild quarterly variation  
- Added 95 % confidence intervals  

---

##  Results & Insights
![Netflix Forecast](images/forecast_plot.png)

**Legend**
- **Gray:** Training Data  
- **Blue:** Actual Validation Revenue  
- **Red Dashed:** SARIMAX Forecast  
- **Orange Dashed:** Extended Forecast (8 Quarters Ahead)  
- **Shaded Areas:** 95 % Confidence Intervals  

**Highlights**
- SARIMAX with business drivers cut RMSE from ≈\$661M to ≈\$147M (−77.8%).  
- The forecast projects continued growth toward ≈\$10–11B per quarter by 2026.  
- Confidence intervals expand slightly over time, reflecting natural forecast uncertainty.

---

##  Tech Stack
| Category | Tools |
|-----------|--------|
| Language | Python 3 |
| Libraries | Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib |
| Models | ARIMA, SARIMAX |
| Evaluation | RMSE |
| Visualization | Matplotlib, Seaborn |


