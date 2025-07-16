#!/usr/bin/env python
# coding: utf-8

# # Data Undetstanding

# In[3]:


import pandas as pd


# In[4]:


df=pd.read_csv('/Users/xiaoming/Library/CloudStorage/OneDrive-AsiaPacificUniversity/Degree Y3S2/FYP/retail_store_inventory.csv')


# ##### NORMAL  INFORMATION (ROW+COLUMN AMOUNTS AND ATTRIBUTES DATA TYPES)

# In[5]:


print("ROW, COLUMNS: \n", df.shape)
print("\nATTRIBUTES DATA TYPES: \n", df.dtypes)


# ##### NORMAL DATA INFORMATION (NULL AND DUPLICATES)

# In[6]:


print("\nMISSING DATA: \n",df.isnull().sum())
print("\nDUPLICATES DATA: ",df.duplicated().sum())


# ##### TOTAL UNIQUE VALUE FOR CATEGORY COLUMNS

# In[7]:


print("UNIQUE VALUE:")
for col in ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']:
    print(f"{col}: {df[col].nunique()} unique values")


# ##### RANGE OF NUMERIC ATTRIBUTES

# In[8]:


num_cols = ['Inventory Level', 'Units Sold', 'Units Ordered',
            'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing',]
print("\nNumerical Columns Stats:")
print(df[num_cols].describe())


# In[9]:


print(df.head())


# In[10]:


df["Date"] = pd.to_datetime(df["Date"])
print(df.dtypes)


# In[11]:


import matplotlib.pyplot as plt

# Plot histogram for 'Units Sold'
df['Units Sold'].hist(bins=30, edgecolor='black')
plt.title('Histogram of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.show()


# For a particular column
skewness = df['Units Sold'].skew()
print(f"Skewness: {skewness}")


# #

# #

# #

# # Model Building

# ##### Libraries needed for model building

# In[12]:


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# ##### Lagging

# In[13]:


df['lag_1'] = df['Units Sold'].shift(1)  # Previous day's sales
df['lag_7'] = df['Units Sold'].shift(7)  # Sales from 7 days ago


# ##### Rolling

# In[14]:


df['rolling_mean_7'] = df['Units Sold'].rolling(window=7).mean()  # 7-day rolling mean


# ##### Date Extract

# In[15]:


df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # Weekend flag


# In[16]:


# Assuming 'Category' is an ordinal feature
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])


# In[17]:


# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Store ID', 'Product ID', 'Region', 'Weather Condition','Category','Seasonality'])


# In[18]:


# Add a new column for price difference
df['Price_Difference'] = df['Price'] - df['Competitor Pricing']


# In[19]:


# Define the features and target
X = df.drop(columns=['Units Sold','Date','Units Ordered','Demand Forecast']) 
y = df['Units Sold']


# In[20]:


# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[21]:


df.dtypes


# In[22]:


# Initialize the model
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    eval_metric='rmse',            # RMSE evaluation metric
    n_estimators=100,              # Number of boosting rounds
    max_depth=6,                   # Depth of the trees
    learning_rate=0.1,             # Learning rate
    random_state=42                # For reproducibility
)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


# In[23]:


threshold = 200

y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)


# In[24]:


from sklearn.metrics import confusion_matrix, classification_report, auc
import matplotlib.pyplot as plt

threshold = 200

y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred_class)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))


# In[25]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Units Sold")
plt.show()


# In[26]:


errors = y_test.values - y_pred
plt.hist(errors, bins=50)
plt.title("Prediction Errors")
plt.show()


# In[27]:


import xgboost as xgb
xgb.plot_importance(model)


# In[31]:


model.save_model("xgb_model.json")  # or use .bin

