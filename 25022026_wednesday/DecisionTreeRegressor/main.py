import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

class RegressorModel:

    def __init__(self, path):
        self.df = None
        self.path = path
        self.preprocessor = None
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder(drop = "first")
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.3
        self.random_state = 42
        self.model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, ccp_alpha = 0.01, random_state = self.random_state)
        self.pipeline = None
        self.y_pred = None

    def load_dataset(self):
        """
        Loading the dataset.
        Returns - None
        -------
        """

        self.df = pd.read_csv(self.path)
        print(f"DataFrame = \n\n{self.df.head()}")

        print("\n-----> Dataset Loaded Successfully.")

    def check_dataset(self):
        """
        Checking the dataset.
        Returns - None
        -------
        """

        print("DataFrame Info = \n")
        self.df.info()

        print(f"\nDataFrame Description = \n\n{self.df.describe()}\n")

        print(f"Number of Duplicates in DataFrame = {self.df.duplicated().sum()}")

        print("\n-----> Dataset Checked Successfully.")

    def drop_duplicates(self):
        """
        Dropping duplicates.
        Returns - None
        -------
        """

        self.df.drop_duplicates(inplace = True)
        print(f"Number of Duplicates in DataFrame after Dropping Duplicates = {self.df.duplicated().sum()}")

        print("\n-----> Duplicates Dropped Successfully.")

    def eda(self):
        """
        Performing EDA
        Returns - None
        -------
        """

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        cols = ["bmi", "charges"]
        for i in range(len(cols)):
            plt.subplot(1, 2, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        for i in range(len(cols)):
            plt.subplot(1, 2, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
            plt.title(cols[i])
        plt.show()

        print("-----> EDA Performed Successfully.")

    def handle_outliers(self):
        print("Outliers Info = \n")

        cols = ["bmi", "charges"]
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            print(f"Feature ( {col} ) = {len(outliers)} outliers found.")

        self.df["bmi"] = np.log(self.df["bmi"])
        self.df["charges"] = np.log(self.df["charges"])

        for i in range(len(cols)):
            plt.subplot(1, 2, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        print("\n-----> Outliers Handled Successfully.")

    def encoding(self):
        """
        Encoding the data.
        Returns - None
        -------
        """

        ohe_cols = ["sex", "smoker", "region"]
        self.preprocessor = ColumnTransformer(transformers = [("OHE", self.ohe, ohe_cols)], remainder = "passthrough")

        print("-----> Encoded Successfully.")

    def split_data(self):
        """
        Splitting the data.
        Returns - None
        -------
        """

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.random_state)

        print("-----> Split Data Successfully.")

    def train_model(self):
        """
        Training the model.
        Returns
        -------
        """

        self.pipeline = Pipeline(steps = [("preprocessor", self.preprocessor), ("regressor", self.model)])
        self.pipeline.fit(self.X_train, self.y_train)

        print("-----> Trained Model Successfully.")

    def check_model(self):
        self.y_pred = self.pipeline.predict(self.X_test)

        print(f"MAE = {mean_absolute_error(self.y_test, self.y_pred)}")
        print(f"MSE = {mean_squared_error(self.y_test, self.y_pred)}")
        print(f"R-MSE = {root_mean_squared_error(self.y_test, self.y_pred)}")
        print(f"R2-SCORE = {r2_score(self.y_test, self.y_pred)}")

        print("\n-----> Model Checked Successfully.")

def main():
    print("=========================================== DECISION TREE REGRESSOR ===========================================")
    regressor = RegressorModel("insurance.csv")

    print("\n================================================= LOAD DATASET ================================================\n")
    regressor.load_dataset()

    print("\n================================================ CHECK DATASET ================================================\n")
    regressor.check_dataset()

    print("\n=============================================== DROP DUPLICATES ===============================================\n")
    regressor.drop_duplicates()

    print("\n===================================================== EDA =====================================================\n")
    regressor.eda()

    print("\n=============================================== HANDLE OUTLIERS ===============================================\n")
    regressor.handle_outliers()

    print("\n============================================== ENCODING FEATURES ==============================================\n")
    regressor.encoding()

    print("\n================================================ SPLITTING DATA ===============================================\n")
    regressor.split_data()

    print("\n================================================ TRAINING MODEL ===============================================\n")
    regressor.train_model()

    print("\n================================================== CHECK MODEL ================================================\n")
    regressor.check_model()

    print("\n===============================================================================================================")

if __name__ == "__main__":
    main()