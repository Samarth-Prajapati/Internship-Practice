import pyodbc

print(pyodbc.drivers())

conn = (
    "DRIVER={SQL Server};"
    "SERVER=localhost;"
    "DATABASE=TUTORIAL_1;"
)

try:
    conn = pyodbc.connect(conn)
    print("Connection Successful.")
except ConnectionError:
    print("Connection Error")
except Exception as e:
    print(e)

cursor = conn.cursor()

# query1 = "INSERT INTO TEST1 VALUES (3,'HARSH',21), (4,'ALOK',21), (5,'KAMLESH',22), (6,'YASH',21);"
# cursor.execute(query1)
# conn.commit()

query2 = "SELECT * FROM TEST1;"
cursor.execute(query2)
row = cursor.fetchall()

for i in row:
    print(i)

# query3 = "UPDATE TEST1 SET NAME='SAM' WHERE NAME='SAMARTH';"
# cursor.execute(query3)
# conn.commit()

# query4 = "DELETE FROM TEST1 WHERE NAME='YASH';"
# cursor.execute(query4)
# conn.commit()

cursor.close()
conn.close()