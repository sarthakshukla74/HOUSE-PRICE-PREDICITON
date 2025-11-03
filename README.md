🏠 California Housing Price Prediction

This project performs Exploratory Data Analysis (EDA), data preprocessing, and linear regression modeling on the California Housing Dataset from Scikit-learn.
It demonstrates a full end-to-end Machine Learning workflow — from loading and cleaning data to training, evaluating, and saving a regression model.

📋 Table of Contents

About the Dataset

Workflow Overview

Exploratory Data Analysis (EDA)

Data Preprocessing

Model Training and Evaluation

Results

Model Saving and Loading

Libraries Used

How to Run

📊 About the Dataset

The California Housing Dataset contains data collected from the 1990 U.S. Census, describing housing in California districts.
Each row represents one district, with features describing the demographics, geography, and housing characteristics.

Features:

MedInc: Median income in block group

HouseAge: Median house age

AveRooms: Average number of rooms per household

AveBedrms: Average number of bedrooms per household

Population: Block group population

AveOccup: Average household size

Latitude: Block group latitude

Longitude: Block group longitude

Target:

Price: Median house value (in $100,000s)

⚙️ Workflow Overview

Load Dataset – Using fetch_california_housing()

Convert to DataFrame – Add target column “Price”

EDA – Explore relationships, distributions, and correlations

Outlier Handling –

Capping (using IQR method)

Removing outliers

Normalization / Standardization

Train-Test Split (70% training, 30% testing)

Model Training using Linear Regression

Prediction & Evaluation – MSE, MAE, R², Adjusted R², RMSE

Save & Load Model using pickle

🔍 Exploratory Data Analysis (EDA)

Performed using Seaborn and Matplotlib:

Correlation matrix and heatmap

Pairplot visualization

Boxplots for outlier detection

Distribution of residuals

Example Visuals:

sns.heatmap(data=df.corr(), annot=True)
sns.pairplot(df)
sns.boxplot(data=df)

🧹 Data Preprocessing
1. Handling Outliers

Used the Interquartile Range (IQR) method to cap or remove outliers:

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


Created two datasets:

df_capped → Capped outliers to boundaries

df_cleaned → Removed outliers completely

2. Normalization

Scaled feature values between 0 and 1 using MinMaxScaler.

3. Standardization

Used StandardScaler to standardize training and testing data:

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

🤖 Model Training and Evaluation
Algorithm Used:

Linear Regression (from sklearn.linear_model)

Training:
model = LinearRegression()
model.fit(X_train_norm, Y_train)

Predictions:
reg_pred = model.predict(X_test_norm)

Residuals:

Visualized to check normal distribution:

sns.displot(residuals, kind='kde')

📈 Results
Metric	Description	Value
MSE	Mean Squared Error	mse
MAE	Mean Absolute Error	mae
R² Score	Coefficient of determination	r2
Adjusted R²	Adjusted for number of predictors	adjusted_r2
RMSE	Root Mean Squared Error	rmse
💾 Model Saving and Loading
Save Model:
import pickle
pickle.dump(model, open('reg_model.pkl', 'wb'))

Load Model:
model = pickle.load(open('reg_model.pkl', 'rb'))

🧠 Libraries Used
Library	Purpose
pandas	Data manipulation and analysis
numpy	Numerical computations
seaborn	Visualization and EDA
matplotlib	Plotting graphs
scikit-learn	ML algorithms, metrics, preprocessing
pickle	Saving and loading model
▶️ How to Run

Install dependencies

pip install pandas numpy seaborn matplotlib scikit-learn


Run the script

python housing_regression.py


Check model performance metrics printed in the terminal.

The trained model will be saved as reg_model.pkl.

📘 Author

Sarthak Shukla
💻 Data Science & ML Enthusiast
