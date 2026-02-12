from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, StrictStr, StrictInt, StrictFloat, Field
from typing import List

app = FastAPI()

def connect_mongodb():
    """
    Connect to MongoDB
    Returns - collection
    -------
    """

    try:
        client = MongoClient("mongodb://localhost:27017/")
        print("Connected to MongoDB")

        database = client.get_database("Tutorial_1")
        collection = database["Test2"]
        return collection

    except ConnectionError as error:
        print("Connection Error - ", error)

    except Exception as error:
        print(error)

class Student(BaseModel):
    name: StrictStr
    age: StrictInt = Field(gt = 0, lt = 25)
    marks: List[StrictFloat] = List[Field(gt = -1, lt = 25)]

@app.post("/add_students", status_code = status.HTTP_201_CREATED)
def add_students(request: Student):
    """
    Add students
    Parameters
    ----------
    request : Student --> BaseModel

    Returns - Added message
    -------

    """

    db = connect_mongodb()

    total_students = [i for i in db.find()]
    _id = len(total_students) + 1

    if db.count_documents({"name": request.name}) > 0:
        raise HTTPException(status_code = 400, detail = "Student already exists")

    db.insert_one(
        {
            "_id": _id,
            "name": request.name,
            "age": request.age,
            "marks": request.marks,
        }
    )
    return {"message": f"ID {_id} - Student( {request.name} ) Added Successfully"}

@app.get("/students", status_code = status.HTTP_200_OK)
def get_students():
    """
    Get students
    Parameters
    ----------
    request - Student --> BaseModel

    Returns - all student details
    -------

    """

    db = connect_mongodb()

    total_students = [i for i in db.find()]
    return total_students

@app.get("/students/{_id}", status_code = status.HTTP_200_OK)
def get_students_by_id(_id: int):
    """
    Get student details by ID
    Parameters
    ----------
    _id - Student ID

    Returns - student detail
    -------
    """

    db = connect_mongodb()

    student = db.find_one({"_id": _id})
    return student

@app.delete("/students/{_id}", status_code = status.HTTP_204_NO_CONTENT)
def delete_student(_id: int):
    """
    Delete student by ID
    Parameters
    ----------
    _id - Student ID

    Returns - Deleted Student Message
    -------
    """

    db = connect_mongodb()

    db.delete_one({"_id": _id})
    return {"message": "Student Deleted Successfully"}

@app.patch("/students/{_id}", status_code = status.HTTP_202_ACCEPTED)
def update_student(_id: int, request: Student):
    """
    Update student details by ID
    Parameters
    ----------
    _id - Student ID
    request - Student --> BaseModel

    Returns - Updated Student Message
    -------
    """

    db = connect_mongodb()

    student = db.find_one({"_id": _id})
    if not student:
        raise HTTPException(status_code = 400, detail = "Student does not exist")

    db.update_one(
        {"_id": _id},
        {"$set":
             {
                 "name": request.name if request.name != "string" else db.find_one({"_id": _id})["name"],
                 "age": request.age if request.age != 1 else db.find_one({"_id": _id})["age"],
             },
            "$push":
            {
                "marks": request.marks if len(request.marks) > 1 or request.marks[0] != 0 else db.find_one({"_id": _id})["marks"],
            }
         },
    )

    return {"message": "Student Updated Successfully"}

