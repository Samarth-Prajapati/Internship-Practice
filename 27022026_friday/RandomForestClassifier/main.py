import pandas as pd

seperator = f"\n{'--' * 70}\n"

class Classification:

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
        print(f"\nDataset = \n{self.df.head()}")
        print("\nDataset Loaded Successfully.", end = seperator)

    def check_data(self):
        """
        Check if data is ready for model training
        Returns - None
        -------
        """

        print("Dataset Info = ")
        self.df.info()
        print(f"\nDataset Description = \n{self.df.describe()}\n")
        print(f"Dataset Shape = {self.df.shape}")
        print(f"Total Duplicates in Dataset = {self.df.duplicated().sum()}")
        print(f"\nTotal Null Values in Dataset = \n{self.df.isnull().sum()}\n")
        print("Dataset Checked Successfully.", end = seperator)

def main():
    path = "news.csv"
    classifier = Classification(path)

    classifier.load_data()
    classifier.check_data()

if __name__ == "__main__":
    main()