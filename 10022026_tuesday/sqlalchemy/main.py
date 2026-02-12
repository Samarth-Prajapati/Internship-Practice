from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

conn = "mssql+pyodbc://localhost/TUTORIAL_1?driver=ODBC+Driver+17+for+SQL+Server"

try:
    engine = create_engine(conn)
    print("Connection Successful")

    Base = declarative_base()
    print("Base Created")

    Session = sessionmaker(bind = engine)
    session = Session()
    print("Session Created")
except ConnectionError:
    print("Connection Error")
except Exception as e:
    print(e)

class Test2(Base):
    __tablename__ = "TEST2"
    id = Column(Integer, primary_key = True, autoincrement = True)
    name = Column(String)
    age = Column(Integer)

try:
    Base.metadata.create_all(engine)
except Exception as e:
    print(e)

def get_db():
    db = session
    try:
        yield db
    except Exception as e:
        print(e)
    finally:
        db.close()