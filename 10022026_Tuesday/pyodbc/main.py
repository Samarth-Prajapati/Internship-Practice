import pyodbc

print(pyodbc.drivers())

conn = (
    'DRIVER={SQL Server};'
    'SERVER=localhost;'
    'DATABASE=TUTORIAL_1;'
)

try:
    conn = pyodbc.connect(conn)
    print("Success! You are connected.")
except Exception as e:
    print(f"Still not working? Here is why: {e}")
cursor = conn.cursor()

cursor.execute("SELECT * FROM TEST1")
row = cursor.fetchall()
print(row)

cursor.close()
conn.close()