import pandas as pd

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

        print("DataFrame Info = \n\n")
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
        pass

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

if __name__ == "__main__":
    main()