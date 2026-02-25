import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class RegressorModel:

    def __init__(self):
        self.df = None

    def load_dataset(self):
        """
        Loading the dataset.
        Returns - None
        -------
        """

        self.df = pd.read_csv("insurance.csv")
        print(f"DataFrame = \n\n{self.df.head()}")

        print("\n[INFO] Dataset Loaded Successfully.")

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

        print("\n[INFO] Dataset Checked Successfully.")

    def drop_duplicates(self):
        """
        Dropping duplicates.
        Returns - None
        -------
        """

        self.df.drop_duplicates(inplace = True)
        print(f"Number of Duplicates in DataFrame after Dropping Duplicates = {self.df.duplicated().sum()}")

        print("\n[INFO] Duplicates Dropped Successfully.")

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

        print("[INFO] EDA Performed Successfully.")

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

        print("\n[INFO] Outliers Handled Successfully.")

def main():
    print("=========================================== DECISION TREE REGRESSOR ===========================================")
    regressor = RegressorModel()

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

if __name__ == "__main__":
    main()