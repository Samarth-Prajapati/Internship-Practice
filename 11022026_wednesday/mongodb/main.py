from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

def setup_mongodb():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS = 2000)
        client.server_info()
        print("Connection Successful")

        return client

    except ConnectionError:
        print("Connection Error")

    except ServerSelectionTimeoutError:
        print("Server Selection Timeout Error")

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

def list_collections(database):
    collections = database.list_collection_names()
    if not collections:
        print("No Collections Found")
        return None

    return collections

def find_document(collection, data):
    result = collection.find_one(data)
    if not result:
        print("Document Not Found")
        return None

    return result

def all_documents(collection):
    result = collection.find()
    if not result:
        print("Document Not Found")
        return None

    for person in result:
        print(f"Name : {person["name"]}, Role : {person["role"]}, Skills : {person["skills"]}")
    return "Fetched Successfully"

def delete_document(collection, data):
    data_delete = find_document(collection, data)
    if not data_delete:
        print("Document Not Found")
        return None

    collection.delete_one(data)
    return "Deleted Successfully"

def update_document(collection, identifier, updated_data):
    updated = collection.update_one(
        identifier,
        {"$set" : updated_data}
    )
    if not updated:
        print("Document Not Found")
        return None

    return updated

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

# list_of_collections = list_collections(db)
# print(list_of_collections)

data3 = {
    "name" : "Samarth Prajapati",
}

# find_data = find_document(table, data3)
# print(find_data)

data4 = {
    "name" : "Om Mishra",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

data5 = {
    "name" : "Alok Yadav",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

data6 = {
    "name" : "Harsh Mistry",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

data7 = {
    "name" : "Kamlesh Patel",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

data8 = {
    "name" : "Yash Maisuriya",
    "role" : "AI Intern",
    "skills" : ["Python", "AI"]
}

# document3 = add_document(table, data4)
# document4 = add_document(table, data5)
# document5 = add_document(table, data6)
# document6 = add_document(table, data7)
# document7 = add_document(table, data8)

# all_docs = all_documents(table)
# print(all_docs)

data9 = {
    "name" : "Yash Maisuriya",
}

# del_document = delete_document(table, data9)
# print(del_document)

identifier_data = {
    "name" : "Alok Yadav"
}

update_data = {
    "role" : "BA"
}

# update = update_document(table, identifier_data, update_data)
