# 🏡 California Housing Price Prediction

> 📊 *An end-to-end Machine Learning project demonstrating Exploratory Data Analysis (EDA), Outlier Handling, Data Normalization, Linear Regression Modeling, and Model Evaluation on the California Housing Dataset.*

---

## 🧠 Overview  

This project predicts **median house prices** in California using multiple features such as income, house age, population, and geographical location.  
It showcases a complete **Machine Learning pipeline**, from **data loading → EDA → preprocessing → model training → evaluation → saving the model**.

---

## 📦 Dataset Details  

**Dataset:** California Housing Dataset (from `sklearn.datasets`)  
**Description:** Derived from the 1990 U.S. Census — each record represents a California district.

| Feature | Description |
|----------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household size |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| **Target → `Price`** | Median house value (in $100,000s) |

---

## ⚙️ Project Workflow  

### 🧾 1. **Load and Inspect Data**
- Fetch data using `fetch_california_housing()`
- Convert to Pandas DataFrame
- Add the target column `Price`
- Display dataset info and summary statistics  

---

### 🔍 2. **Exploratory Data Analysis (EDA)**  
Visualizations and relationships explored using **Seaborn** and **Matplotlib**.

📌 **Key Steps:**
- Correlation matrix & heatmap  
- Pairplots for feature relationships  
- Boxplots for outlier detection  
- Distribution of residuals  

📈 Example:
```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
sns.pairplot(df)
sns.boxplot(data=df)
```

---

### 🧹 3. **Data Preprocessing**

#### 🧾 Outlier Handling  
- **Capping** using IQR limits  
- **Outlier Removal** for clean dataset

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

Created:
- `df_capped` → Outliers replaced with boundary values  
- `df_cleaned` → Outliers completely removed  

#### 🔄 Normalization & Standardization  
- Normalized data with **MinMaxScaler**  
- Standardized train/test data with **StandardScaler**

---

### 🧩 4. **Train-Test Split**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```

---

### 🤖 5. **Model Training**
Trained a **Linear Regression Model** using standardized data.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_norm, Y_train)
```

---

### 🔮 6. **Predictions & Residuals**
Generated predictions and analyzed residuals for normal distribution.

```python
reg_pred = model.predict(X_test_norm)
sns.displot(Y_test - reg_pred, kind='kde')
```

---

### 🧮 7. **Model Evaluation**

| Metric | Description | Formula / Purpose |
|--------|--------------|-------------------|
| **MSE** | Mean Squared Error | Measures average squared error |
| **MAE** | Mean Absolute Error | Measures average absolute difference |
| **R² Score** | Coefficient of determination | Goodness of fit |
| **Adjusted R²** | Adjusted for number of predictors | Penalizes extra features |
| **RMSE** | Root Mean Squared Error | Interpretable version of MSE |

Example:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(Y_test, reg_pred)
mae = mean_absolute_error(Y_test, reg_pred)
r2 = r2_score(Y_test, reg_pred)
```

📊 **Adjusted R² Calculation:**
```python
n = len(Y_test)
p = X_test_norm.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

### 💾 8. **Saving & Loading the Model**

✅ **Save Model:**
```python
import pickle
pickle.dump(model, open('reg_model.pkl', 'wb'))
```

✅ **Load Model:**
```python
model = pickle.load(open('reg_model.pkl', 'rb'))
```

---

## 📈 Results Summary  

| Metric | Value (Example) |
|---------|----------------|
| MSE | *e.g., 0.54* |
| MAE | *e.g., 0.47* |
| R² Score | *e.g., 0.74* |
| Adjusted R² | *e.g., 0.73* |
| RMSE | *e.g., 0.73* |

> 🧩 *The Linear Regression model performs reasonably well with a strong R² score, indicating a good fit between predicted and actual housing prices.*

---

## 🧰 Libraries Used  

| Library | Purpose |
|----------|----------|
| **pandas** | Data manipulation |
| **numpy** | Numerical computations |
| **seaborn** | Visualization and EDA |
| **matplotlib** | Plotting graphs |
| **scikit-learn** | ML models, preprocessing, metrics |
| **pickle** | Saving and loading models |

---

## 🚀 How to Run the Project  

1. **Install Required Libraries**
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

2. **Run the Python Script**
   ```bash
   python housing_regression.py
   ```

3. **Check Outputs**
   - Model metrics (MSE, MAE, R², etc.)
   - Saved model file: `reg_model.pkl`

4. **Load and Use Model**
   ```python
   import pickle
   model = pickle.load(open('reg_model.pkl', 'rb'))
   ```

---

## 👨‍💻 Author  

**👋 Sarthak Shukla**  
📍 Sophomore, B.Tech CSE (DS) — PSIT Kanpur  
💡 Passionate about Machine Learning & Data Science  
🔗 [LinkedIn](https://www.linkedin.com/in/sarthak-shukla) | [LeetCode](https://leetcode.com)  

---

✨ *“Data is the new oil, but insight is the new currency.”*
