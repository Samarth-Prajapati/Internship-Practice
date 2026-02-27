import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

seperator = f"\n{'-' * 70}\n"

class DataBase:

    def __init__(self):
        try:
            self.engine = create_engine("mssql+pyodbc://localhost/ML?driver=ODBC+Driver+17+for+SQL+Server")

        except SQLAlchemyError:
            print("SQLAlchemy Error")
        except Exception as error:
            print(error)

    def load_table(self, table_name):
        """
        Load DB Table to dataframe
        Parameters
        ----------
        table_name - table name

        Returns - dataframe
        -------
        """

        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df

        except Exception as error:
            print(error)

def main():
    loader = DataBase()
    df = loader.load_table("dataset")
    print(df.head())

if __name__ == "__main__":
    main()
