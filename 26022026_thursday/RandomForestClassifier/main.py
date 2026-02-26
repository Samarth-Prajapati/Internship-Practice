import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

class Classification:

    def __init__(self, path):
        self.df = None
        self.path = path
        self.preprocessor = None
        self.le = LabelEncoder()

    def load_data(self):
        """
        Loading dataset.
        Returns - None
        -------
        """

        self.df = pd.read_csv(self.path)
        print(f"DataFrame = \n\n{self.df.head()}")

        print("\n-----> Dataset Loaded Successfully.")

    def check_dataset(self):
        """
        Checking dataset.
        Returns - None
        -------
        """

        print("DataFrame Info = \n")
        self.df.info()

        print(f"\nDataFrame Description = \n\n{self.df.describe()}\n")

        print(f"DataFrame Shape = {self.df.shape}\n")

        print(f"Number of Duplicates in DataFrame = {self.df.duplicated().sum()}")

        print("\n-----> Dataset Checked Successfully.")

    def eda(self):
        """
        Performing EDA
        Returns - None
        -------
        """

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        cols = self.df.columns.tolist()
        cols = cols[:len(cols) - 1]
        for i in range(len(cols)):
            plt.subplot(3, 3, i + 1)
            sns.boxplot(y = self.df[cols[i]])
        plt.show()

        for i in range(len(cols)):
            plt.subplot(3, 3, i + 1)
            sns.histplot(self.df[cols[i]], kde = True)
        plt.show()

        print("-----> EDA Performed Successfully.")

    def encoding(self):
        """
        Encoding the data.
        Returns - None
        -------
        """

        le_cols = ["label"]
        self.preprocessor = ColumnTransformer(transformers = [("LE", self.le, le_cols)], remainder = "passthrough")

        print("-----> Encoded Successfully.")

def main():
    print("========================================== RANDOM FOREST CLASSIFIER ===========================================")
    random_forest = Classification("CropRecommendation.csv")

    print("\n================================================= LOAD DATASET ================================================\n")
    random_forest.load_data()

    print("\n================================================ CHECK DATASET ================================================\n")
    random_forest.check_dataset()

    print("\n===================================================== EDA =====================================================\n")
    random_forest.eda()

    print("\n============================================== ENCODING FEATURES ==============================================\n")
    random_forest.encoding()

if __name__ == "__main__":
    main()