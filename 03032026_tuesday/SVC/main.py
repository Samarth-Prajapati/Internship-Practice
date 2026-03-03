import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

seperator = f"\n\n{'--' * 70}\n\n"

class SupportVectorClassifier:
    """
    Support Vector Machine Classifier Implementation
    """

    def __init__(self, path):
        self.path = path
        self.df = None
        self.encoder = LabelEncoder()

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
        Returns - None
        -------
        """

        print("Dataset Info = ")
        self.df.info()

        print(f"\nDataset Description = \n{self.df.describe()}\n")
        print(f"Total Duplicate Values = {self.df.duplicated().sum()}")
        print(f"Dataset Shape = {self.df.shape}")
        print(f"\nTotal Null Values = \n{self.df.isnull().sum()}\n")

        print("Data Checked Successfully.", end = seperator)

    def preprocess_data(self):
        """
        Preprocess data
        Returns - None
        -------
        """

        self.df.drop("user_id", axis = 1, inplace = True)
        self.df["gender"] = self.encoder.fit_transform(self.df["gender"])
        print(f"Dataset = \n{self.df.head()}\n")

        print("Data Preprocessed Successfully.", end = seperator)

    def eda(self):
        """
        Perform EDA on data
        Returns - None
        -------
        """

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        sns.pairplot(self.df, hue = "purchased")
        plt.show()

        cols = ["gender", "age", "estimated_salary"]
        for i in range(len(cols)):
            plt.subplot(1, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
            plt.title(cols[i])
        plt.show()

        for i in range(len(cols)):
            plt.subplot(1, 3, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
            plt.title(cols[i])
        plt.show()

        print("EDA performed Successfully.", end = seperator)

def main():
    svc = SupportVectorClassifier("user_data.csv")
    svc.load_data()
    svc.check_data()
    svc.preprocess_data()
    svc.eda()

if __name__ == "__main__":
    main()