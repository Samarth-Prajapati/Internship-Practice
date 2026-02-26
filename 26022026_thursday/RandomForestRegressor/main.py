import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

seperator = f"\n\n{'--' * 75}\n\n"

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    filename = 'log_regressor.log',
    filemode = 'w'
)

class Regressor:
    """
    PERFORMING RANDOM FOREST REGRESSOR MODEL ON FLIGHT PRICE PREDICTION DATASET
    """

    def __init__(self, path):
        self.path = path
        self.df = None
        self.logger = logging.getLogger("RANDOM_FOREST_REGRESSOR")

    def load_dataset(self):
        """
        LOADING DATASET
        Returns - None
        -------
        """

        try:
            self.logger.info("Start Loading Dataset")

            self.df = pd.read_csv(self.path)
            print(f"\nDataset = \n\n{self.df.head()}\n")

            print("------> Dataset Loaded Successfully", end = seperator)
            self.logger.info("Dataset Loaded Successfully")

        except FileNotFoundError:
            print("File Not Found")
            self.logger.error("File Not Found")

        except Exception as error:
            print(error)
            self.logger.error(error)

    def check_dataset(self):
        """
        CHECKING DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Analyzing Dataset")

        print("Dataset Info = \n")
        self.df.info()

        print(f"\nDataset Description = \n\n{self.df.describe()}\n")
        print(f"Shape of Dataset = {self.df.shape}\n")
        print(f"Sum of Duplicates = {self.df.duplicated().sum()}\n")

        print("------> Dataset Analysis Successfully", end=seperator)
        self.logger.info("Dataset Analysis Successfully")

    def preprocess(self):
        """
        PREPROCESSING DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Preprocessing Dataset")

        self.df.drop(self.df.columns[0], axis = 1, inplace = True)
        print("Removed Unwanted Columns\n")
        print(f"Dataset = \n\n{self.df.head()}\n")

        print("------> Dataset Preprocessed Successfully", end=seperator)
        self.logger.info("Dataset Preprocessed Successfully")

    def eda(self):
        """
        PERFORMING EDA ON DATASET
        Returns - None
        -------
        """

        self.logger.info("Start Performing EDA on Dataset")

        corr = self.df.corr(numeric_only = True)
        sns.heatmap(corr, annot = True)
        plt.show()

        cols = ["duration", "days_left", "price"]
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

        print("------> EDA Performed Successfully", end=seperator)
        self.logger.info("EDA Performed Successfully")

def main():
    """
    ALL OPERATIONS OF REGRESSOR CLASS ARE COMPLETED HERE
    Returns - None
    -------
    """

    path = "flight_price_prediction.csv"
    regressor = Regressor(path)

    regressor.load_dataset()
    regressor.check_dataset()
    regressor.preprocess()
    regressor.eda()

if __name__ == "__main__":
    main()