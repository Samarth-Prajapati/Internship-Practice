import pandas as pd

seperator = f"\n\n{'--' * 75}\n\n"

class ClusteringAlgorithm:
    """Implementation of K Means Clustering Algorithm"""

    def __init__(self):
        self.df = None

    def load_dataset(self):
        """
        Loading .csv file to dataframe
        Returns - None
        -------
        """

        try:
            self.df = pd.read_csv("MallCustomers.csv")
            print(f"\nDataset = \n{self.df.head()}\n")

            print("Dataset Loaded Successfully.", end = seperator)

        except FileNotFoundError:
            print("File not found.")
        except Exception as error:
            print(error)

    def analyse_dataset(self):
        """
        Analysis of dataset to check null values and duplicates, necessary for preprocessing data
        Returns - None
        -------
        """

        self.load_dataset()

        try:
            print("Dataset Info = ")
            self.df.info()

            print(f"\nDataset Description = \n{self.df.describe()}\n")
            print(f"Checking Null Values = \n{self.df.isnull().sum()}\n")
            print(f"Sum of Duplicates = {self.df.duplicated().sum()}")
            print(f"Dataset Shape = {self.df.shape}\n")

            print("Dataset Analysis Successful.", end = seperator)

        except Exception as error:
            print(error)

    def preprocess_dataset(self):
        """
        Preprocessing dataset by dropping unnecessary columns
        Returns - None
        -------
        """

        self.analyse_dataset()

        try:
            self.df.drop("CustomerID", axis = 1, inplace = True)
            print(f"After dropping unnecessary features dataset = \n{self.df.head()}\n")

            print("Dataset Preprocessing Successful.", end = seperator)

        except Exception as error:
            print(error)

def main():
    """Main Function"""

    cluster = ClusteringAlgorithm()
    cluster.preprocess_dataset()

if __name__ == "__main__":
    main()