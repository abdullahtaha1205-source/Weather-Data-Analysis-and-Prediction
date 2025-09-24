import pandas as pd import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression from sklearn.metrics import mean_squared_error, r2_score

Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv" df = pd.read_csv(url)

print("Dataset Sample:") print(df.head())

Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

Feature Engineering
df['Year'] = df['Date'].dt.year df['Month'] = df['Date'].dt.month df['Day'] = df['Date'].dt.day

Features (X) and Target (y)
X = df[['Year', 'Month', 'Day']] y = df['Temp']

Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train model
model = LinearRegression() model.fit(X_train, y_train)

Predictions
y_pred = model.predict(X_test)

Evaluation
print("\nModel Performance:") print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) print("R2 Score:", r2_score(y_test, y_pred))

Plot actual vs predicted
plt.figure(figsize=(8,5)) plt.scatter(y_test, y_pred, alpha=0.5) plt.xlabel("Actual Temperature") plt.ylabel("Predicted Temperature") plt.title("Actual vs Predicted Temperatures") plt.show()

Custom Prediction Example
sample = pd.DataFrame({'Year':[1991], 'Month':[6], 'Day':[15]}) print("\nPredicted Temp on 1991-06-15:", model.predict(sample)[0])
