# Importing Dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load .csv File to DataFrame
print("Insurance DataFrame - ")

df = pd.read_csv("../19022026_thursday/insurance.csv")
print(df.head(), "\n")

print(f"Info -\n{df.info()}\n")
print(f"Description -\n{df.describe()}\n")

print(f"Number of Duplicates = {df.duplicated().sum()}")
print("\n")
print(f"Duplicate Row = \n{df[df.duplicated()]}\n")

# Dropping Duplicates
df.drop_duplicates(keep = "first", inplace = True)
print(f"After Dropping Duplicate Rows, Number of Duplicates = {df.duplicated().sum()}\n")

# Histogram
plt.figure(figsize = (6, 4))
sns.histplot(df["charges"], kde = True)
plt.title("Charges")
plt.show()

# One Hot Encoding
print("Transforming Categorical Data to Spare Matrix using OHE - ")
df = pd.get_dummies(df, drop_first = True).astype(int)
print(df.head(), "\n")

# Target Variable
y = df["charges"]
print(f"Target Variable Shape - {y.shape}\n")

# Feature Variables
X = df.drop("charges", axis = 1)
print(f"Feature Variables Shape - {X.shape}\n")

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(f"After Splitting Data -\nX_train shape = {X_train.shape}\nX_test shape = {X_test.shape}\ny_train shape = {y_train.shape}\ny_test shape = {y_test.shape}\n")

# Standard Scaling -> to scale data with mean = 0 & sd = 1
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
print(f"After Scaling Feature Variables - \n{X_train.shape}\n{X_test.shape}\n")

# Training Model
print("Model Training -")
model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_predict)
print(f"\nMAE = {mae}\n")

# Mean Squared Error
mse = mean_squared_error(y_test, y_predict)
print(f"MSE = {mse}\n")

# Root Mean Squared Error
r_mse = root_mean_squared_error(y_test, y_predict)
print(f"R-MSE = {r_mse}\n")

# R2 Score
r2 = r2_score(y_test, y_predict)
print(f"R2 SCORE = {r2}\n")

print(f"Intercept = {model.intercept_}\n")
print(f"Coefficient = {model.coef_}")