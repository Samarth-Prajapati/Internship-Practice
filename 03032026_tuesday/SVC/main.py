import pandas as pd

seperator = f"\n\n{'--' * 70}\n\n"

class SupportVectorClassifier:
    """
    Support Vector Machine Classifier Implementation
    """

    def __init__(self, path):
        self.path = path
        self.df = None

    def load_data(self):
        """
        Load data from csv file
        Returns - None
        -------
        """

        self.df = pd.read_csv(self.path)
        print(f"\nDataset = \n{self.df.head()}\n")
        print("Data Loaded Successfully.", end = seperator)

    def check_data(self):
        """
        Check if data is correct for model training
        Returns
        -------
        """

        print("Dataset Info = ")
        self.df.info()
        print(f"\nDataset Description = \n{self.df.describe()}\n")
        print(f"Total Duplicate Values = {self.df.duplicated().sum()}")
        print(f"Dataset Shape = {self.df.shape}")
        print(f"\nTotal Null Values = \n{self.df.isnull().sum()}\n")
        print("Data Checked Successfully.", end = seperator)

def main():
    svc = SupportVectorClassifier("user_data.csv")
    svc.load_data()
    svc.check_data()

if __name__ == "__main__":
    main()