import pandas as pd
import logging

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

        print("Dataset Info = \n")
        self.df.info()

        print(f"\nDataset Description = \n\n{self.df.describe()}\n")

        print(f"Shape of Dataset = {self.df.shape}\n")

        print(f"Sum of Duplicates = {self.df.duplicated().sum()}\n")

        print("------> Dataset Analysis Successfully", end=seperator)
        self.logger.info("Dataset Analysis Successfully")

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

if __name__ == "__main__":
    main()