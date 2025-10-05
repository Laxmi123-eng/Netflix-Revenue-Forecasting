#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install statsmodels')


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# In[11]:


df = pd.read_csv("C:/Users/kprab/OneDrive/Documents/TCD/Projects/Netflix Revenue data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

# Convert and sort
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date")

# Clean revenue column
df["Global_Revenue"] = df["Global_Revenue"].replace('[\$,]', '', regex=True).astype(float)

# Set Date as index (important for ARIMA)
df.set_index("Date", inplace=True)

# Plot revenue
df["Global_Revenue"].plot(figsize=(10,5), title="Netflix Global Revenue (Quarterly)", color="purple")
plt.ylabel("Revenue ($)")
plt.show()


# In[12]:


get_ipython().system('pip install pmdarima')


# In[13]:


train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print("Train samples:", len(train))
print("Test samples:", len(test))


# In[14]:


# Fit SARIMA model
model = SARIMAX(
    train["Global_Revenue"],
    order=(1,1,1),            # ARIMA(p,d,q)
    seasonal_order=(1,1,1,4), # SARIMA(P,D,Q,s)
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

# Forecast
pred = results.get_forecast(steps=len(test))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# Evaluate
rmse = np.sqrt(mean_squared_error(test["Global_Revenue"], pred_mean))
print(f"SARIMA Validation RMSE: ${rmse:,.0f}")


# In[15]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

train_size = int(len(df)*0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# simpler ARIMA without seasonal terms
model = ARIMA(train["Global_Revenue"], order=(1,1,1))
results = model.fit()

pred = results.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test["Global_Revenue"], pred))
print(f"Simplified ARIMA RMSE: ${rmse:,.0f}")

plt.figure(figsize=(10,5))
plt.plot(train.index, train["Global_Revenue"], label="Train")
plt.plot(test.index, test["Global_Revenue"], label="Actual", color="blue")
plt.plot(test.index, pred, "--", color="red", label="Forecast")
plt.title("Netflix Revenue Forecast (ARIMA (1,1,1))")
plt.legend(); plt.tight_layout(); plt.show()


# In[16]:


order=(1,1,1)


# In[17]:


# --- Ensure all columns are numeric ---
exog_cols = ["UCAN_Members", "EMEA__Members", "APAC_Members", "UCAN_ARPU", "EMEA_ARPU"]

# Convert commas, spaces, and dollar signs to pure numbers (if any)
for col in exog_cols + ["Global_Revenue"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("[^0-9.-]", "", regex=True)
        .astype(float)
    )


# In[18]:


train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

train_exog = train[exog_cols]
test_exog = test[exog_cols]

model = SARIMAX(
    train["Global_Revenue"],
    exog=train_exog,
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

pred = results.get_forecast(steps=len(test), exog=test_exog)
pred_mean = pred.predicted_mean

from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(test["Global_Revenue"], pred_mean))
print(f"SARIMAX RMSE: ${rmse:,.0f}")


# In[54]:


# --- 1. Clean & ensure numeric types ---
exog_cols = ["UCAN_Members", "EMEA__Members", "APAC_Members", "UCAN_ARPU", "EMEA_ARPU"]

for col in exog_cols + ["Global_Revenue"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("[^0-9.-]", "", regex=True)  # remove commas, $, spaces
        .astype(float)
    )

# --- 2. Split train/test ---
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
train_exog = train[exog_cols]
test_exog = test[exog_cols]

# --- 3. Fit baseline ARIMA (no exog) ---
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

arima_model = ARIMA(train["Global_Revenue"], order=(1,1,1))
arima_results = arima_model.fit()
arima_pred = arima_results.forecast(steps=len(test))
arima_rmse = np.sqrt(mean_squared_error(test["Global_Revenue"], arima_pred))
print(f"Baseline ARIMA(1,1,1) RMSE: ${arima_rmse:,.0f}")

# --- 4. Fit SARIMAX with regressors ---
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_model = SARIMAX(
    train["Global_Revenue"],
    exog=train_exog,
    order=(1,1,1),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_results = sarimax_model.fit(disp=False)

sarimax_pred = sarimax_results.get_forecast(steps=len(test), exog=test_exog)
sarimax_mean = sarimax_pred.predicted_mean
sarimax_ci = sarimax_pred.conf_int()
sarimax_rmse = np.sqrt(mean_squared_error(test["Global_Revenue"], sarimax_mean))
print(f"SARIMAX (with Members + ARPU) RMSE: ${sarimax_rmse:,.0f}")

# --- 5. Combined plot ---
plt.figure(figsize=(12,6))
plt.plot(train.index, train["Global_Revenue"], label="Train", color="gray")
plt.plot(test.index, test["Global_Revenue"], label="Actual", color="blue")
plt.plot(test.index, arima_pred, "--", label="ARIMA Forecast", color="green")
plt.plot(test.index, sarimax_mean, "--", label="SARIMAX Forecast", color="red")
plt.fill_between(test.index, sarimax_ci.iloc[:,0], sarimax_ci.iloc[:,1], color="red", alpha=0.1)

plt.title("Netflix Quarterly Revenue Forecast — ARIMA vs SARIMAX")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.legend()
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='y')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B'))
plt.ylabel("Revenue ($ Billions)")
plt.show()

# --- 6. Display comparison summary ---
import pandas as pd
summary = pd.DataFrame({
    "Model": ["ARIMA(1,1,1)", "SARIMAX (ARIMA + exog)"],
    "Validation RMSE ($)": [arima_rmse, sarimax_rmse]
})
display(summary)


# In[56]:


# columns used in SARIMAX
exog_cols = ["UCAN_Members","EMEA__Members","APAC_Members","UCAN_ARPU","EMEA_ARPU"]

h = 8  # forecast horizon in quarters

# ---  Create exogenous features with trend + gentle oscillation ---
exog_cols = ["UCAN_Members","EMEA__Members","APAC_Members","UCAN_ARPU","EMEA_ARPU"]
future_rows = []
last = df[exog_cols].iloc[-1].copy()

for i in range(h):
    for col in ["UCAN_Members","EMEA__Members","APAC_Members"]:
        pct = df[col].pct_change().tail(4).mean()
        pct = 0 if np.isnan(pct) else pct
        # add mild random shock ±0.20% and small sine oscillation
        shock = np.random.uniform(-0.020, 0.020)
        seasonal = 0.01 * np.sin(i * np.pi / 2)
        last[col] = last[col] * (1 + pct + shock + seasonal)

    for col in ["UCAN_ARPU","EMEA_ARPU"]:
        delta = df[col].diff().tail(4).mean()
        delta = 0 if np.isnan(delta) else delta
        # add slight noise ±0.1 and mild wave
        last[col] = last[col] + delta + np.random.uniform(-0.1, 0.1) + 0.15*np.sin(i)

    future_rows.append(last.copy())

future_exog = pd.DataFrame(future_rows, columns=exog_cols)
future_index = pd.date_range(start=test.index[-1], periods=h+1, freq="Q")[1:]

# --- 2️⃣ Forecast using the extended exog ---
future_fc = sarimax_results.get_forecast(steps=h, exog=future_exog)
future_mean = future_fc.predicted_mean
future_ci = future_fc.conf_int()

# --- 3️⃣ Plot combined (train, actual, SARIMAX, and extended) ---
plt.figure(figsize=(12,6))
plt.plot(train.index, train["Global_Revenue"], label="Train", color="gray")
plt.plot(test.index, test["Global_Revenue"], label="Actual", color="blue")
plt.plot(test.index, sarimax_mean, "--", color="red", label="SARIMAX Forecast")
plt.plot(future_index, future_mean, "--", color="orange", label="Extended Forecast")
plt.fill_between(future_index, future_ci.iloc[:,0], future_ci.iloc[:,1], color="orange", alpha=0.2)
plt.title("Netflix Revenue Forecast — Extended 8 Quarters with Seasonal Variation")
plt.xlabel("Date"); plt.ylabel("Revenue ($)")
plt.ticklabel_format(style='plain', axis='y')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B'))
plt.ylabel("Revenue ($ Billions)")
plt.legend(); plt.tight_layout(); plt.show()


