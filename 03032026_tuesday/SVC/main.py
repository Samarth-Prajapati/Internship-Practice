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

def main():
    svc = SupportVectorClassifier("user_data.csv")
    svc.load_data()

if __name__ == "__main__":
    main()