from pymongo import MongoClient

def setup_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS = 2000)
        client.server_info()
        print("Connection Successful")

        return client

    except ConnectionError:
        print("Connection Error")

def create_database(client):
    try:
        database = client["Tutorial_1"]
        print("Database Created Successfully")

        return database

    except Exception as e:
        print(e)

def create_collection(database):
    try:
        collection = database["Test1"]
        print("Collection Created Successfully")

        return collection

    except Exception as e:
        print(e)

def add_document(collection, document_data):
    try:
        if collection.count_documents(document_data) > 0:
            print("Cannot Insert Duplicate Document")
            return None

        collection.insert_one(document_data)
        print("Document Added Successfully")

        return collection

    except Exception as e:
        print(e)

mongo = setup_mongodb()
db = create_database(mongo)
table = create_collection(db)

data1 = {
    "name" : "Samarth Prajapati",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

data2 = {
    "name" : "Samarth Prajapati",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

# document1 = add_document(table, data1)
# document2 = add_document(table, data2)
# print(document2)