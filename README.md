#  Netflix Quarterly Revenue Forecasting (Time Series Project)

##  Overview
I developed a **time-series forecasting model** to predict Netflix’s quarterly global revenue using both historical data and regional business drivers (memberships and ARPU).  
The goal was to analyze Netflix’s revenue trend, model its seasonality, and forecast future growth.

I implemented **ARIMA** and **SARIMAX (ARIMA with Exogenous Variables)** models, evaluated performance with RMSE, and extended the forecast 8 quarters into the future.  
This project demonstrates the end-to-end workflow of **data cleaning → model building → evaluation → visualization**.

---

##  Objectives
- Analyze quarterly Netflix revenue trends (2019–2024)
- Build a baseline **ARIMA(1,1,1)** model for pure trend forecasting
- Improve predictions using **SARIMAX** with membership and ARPU features
- Forecast **8 future quarters (2024–2026)** with uncertainty bands
- Visualize and compare model performance

---

##  Methodology
### 1. Data Preparation
- Cleaned and formatted Netflix quarterly data
- Converted all numeric fields (revenue, members, ARPU)
- Created lag and growth rate features
- Split dataset into 80% training and 20% test periods

### 2. Modeling
- **ARIMA(1,1,1)** trained on historical revenue
- **SARIMAX(1,1,1)** trained with regional membership and ARPU
- Forecast accuracy measured via RMSE

### 3. Evaluation
| Model | RMSE (USD) | Notes |
|--------|-------------|-------|
| ARIMA(1,1,1) | ≈ $661M | Baseline model |
| SARIMAX(1,1,1) | ≈ $560M | Improved using exogenous data |

### 4. Extended Forecast
- Forecasted 8 future quarters (2024–2026)
- Added realistic business-driven variation (memberships, ARPU)
- Displayed 95% confidence intervals

---

##  Results & Visualization

![Netflix Forecast](images/forecast_plot.png)

**Legend:**
- **Gray:** Training Data  
- **Blue:** Actual Validation Revenue  
- **Red Dashed:** SARIMAX Forecast  
- **Orange Dashed:** Extended Forecast (8 Future Quarters)  
- **Shaded Areas:** 95% Confidence Intervals  

**Key Insights:**
- SARIMAX effectively captured growth driven by membership and ARPU changes.  
- Forecast indicates continued revenue growth, potentially reaching **$10–11B per quarter by 2026**.  
- Confidence intervals widen slightly, reflecting natural uncertainty.

---

##  Tech Stack
| Category | Tools |
|-----------|--------|
| Language | Python |
| Libraries | Pandas, NumPy, Statsmodels, Scikit-learn, Matplotlib |
| Modeling | ARIMA, SARIMAX |
| Evaluation | RMSE |
| Visualization | Matplotlib, Seaborn |

---

## 📂 Repository Structure

