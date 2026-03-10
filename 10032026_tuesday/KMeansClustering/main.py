import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

seperator = f"\n\n{'--' * 75}\n\n"

class ClusteringAlgorithm:
    """Implementation of K Means Clustering Algorithm"""

    def __init__(self):
        self.df = None
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

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
            print(f"After dropping unnecessary features, Dataset = \n{self.df.head()}\n")

            print("Dataset Preprocessing Successful.", end = seperator)

        except Exception as error:
            print(error)

    def eda(self):
        """
        Performing EDA to check whether dataset contains any outliers
        Returns - None
        -------
        """

        self.preprocess_dataset()

        try:
            corr = self.df.corr(numeric_only = True)
            sns.heatmap(corr, annot = True)
            plt.show()

            numeric_columns = self.df.select_dtypes(include = "int64").columns
            for i in range(len(numeric_columns)):
                plt.subplot(1, 3, i + 1)
                sns.boxplot(self.df[numeric_columns[i]])
            plt.show()

            plt.figure(figsize = (5, 5))
            plt.scatter(self.df.iloc[:, -2], self.df.iloc[:, -1])
            plt.xlabel(self.df.columns[-2])
            plt.ylabel(self.df.columns[-1])
            plt.show()

            print("EDA Performed Successfully.", end = seperator)

        except Exception as error:
            print(error)

    def feature_encoding(self):
        """
        Encoding categorical data to numerical data
        Returns - None
        -------
        """

        self.eda()

        try:
            self.df["Gender"] = self.encoder.fit_transform(self.df["Gender"])
            print(f"After encoding categorical features, Dataset = \n{self.df.head()}\n")

            print("Feature Encoded Successfully.", end = seperator)

        except Exception as error:
            print(error)

    def feature_scaling(self):
        """
        Scale data to prevent overfitting
        Returns - None
        -------
        """

        self.feature_encoding()

        try:
            scaled_df = self.scaler.fit_transform(self.df)
            self.df = pd.DataFrame(scaled_df, columns = self.df.columns)
            print(f"After scaling features, Dataset = \n{self.df.head()}\n")

            print("Feature Scaled Successfully.", end = seperator)

        except Exception as error:
            print(error)

def main():
    """Main Function"""

    cluster = ClusteringAlgorithm()
    cluster.feature_scaling()

if __name__ == "__main__":
    main()