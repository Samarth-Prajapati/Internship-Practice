from flask import Flask, request, jsonify

app = Flask(__name__)

students = {}

@app.route("/add_student", methods = ["POST"])
def add_student():
    if request.method == "POST":
        data = request.get_json()
        if data:
            student_id = len(students) + 1
            students[student_id] = data
            return jsonify(students[student_id]), 201
        return None, 200
    return None, 200

@app.route("/get_student", methods = ["GET"])
def get_student():
    if request.method == "GET":
        return jsonify(students), 200
    return None, 200

@app.route("/remove_student/<int:student_id>", methods = ["DELETE"])
def remove_student(student_id):
    if request.method == "DELETE":
        del students[student_id]
        return "Deleted", 200
    return None, 200

if __name__ == "__main__":
    app.run(debug = True)