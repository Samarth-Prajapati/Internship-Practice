import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    main()