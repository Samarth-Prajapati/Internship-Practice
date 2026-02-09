from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# DB Connection
db_url = os.getenv("DATABASE_CONNECTION_URL")
engine = create_engine(db_url)
session = sessionmaker(bind = engine)
Base = declarative_base()
print("Connection Successful")

def get_db():
    db = session()
    try:
        with db as db_session:
            yield db_session
    except ConnectionError:
        print("Database Error")

class Student(Base):
    __tablename__ = "student"
    name = Column(String)
    age = Column(Integer)
    # marks = Column(List[Float])

Base.metadata.create_all(engine)

data = {}

class StudentPut(BaseModel):
    name: str
    age: int
    # marks: Optional[List[float]] = None

@app.get("/students", tags = ["GET Method"])
async def display_students():
    return data

@app.get("/students/{student_id}", tags = ["GET Method"])
async def get_student(student_id: int):
    if student_id not in data.keys():
        raise HTTPException(status_code = 404, detail = "Student not found")
    return data[student_id]

# @app.post("/students/add", tags = ["POST Method"])
# def add_student(student: StudentPut, db: Session = Depends(get_db)):
#     new_student = StudentPut(name=student.name, age=student.age)
#     db.add(new_student)
#     db.commit()
#     db.refresh(new_student)
#     return new_student

@app.post("/students/add", tags = ["POST Method"])
async def add_student(student: StudentPut):
    student_id = len(data) + 1
    data[student_id] = student
    return data[student_id]

@app.put("/students/update/{student_id}", tags = ["PUT Method"])
async def update_student(student_id: int, student: StudentPut):
    if student_id not in data.keys():
        raise HTTPException(status_code = 404, detail = "Student not found")
    data[student_id] = student
    return data[student_id]

@app.delete("/students/remove/{student_id}", tags = ["DELETE Method"])
async def remove_student(student_id: int):
    if student_id in data.keys():
        del data[student_id]
    else:
        raise HTTPException(status_code = 404, detail = "Student not found.")
    return {f"Student With ID - {student_id} Removed Successfully"}
