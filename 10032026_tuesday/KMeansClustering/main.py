import pandas as pd

seperator = f"\n\n{'----' * 70}\n\n"

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

        self.df = pd.read_csv("MallCustomers.csv")
        print(f"\nDataset = \n{self.df.head()}\n")

        print("Dataset Loaded Successfully.", end = seperator)



def main():
    """Main Function"""

    cluster = ClusteringAlgorithm()
    cluster.load_dataset()

if __name__ == "__main__":
    main()