from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Menyimpan status terakhir di memori (bisa diganti DB kalau mau)
latest_parking_status = None

@app.route('/')
def index_route():

@app.route("/api/parking-status", methods=["POST"])
def receive_parking_status():
    global latest_parking_status
    data = request.get_json()

    latest_parking_status = {
        "received_at": datetime.utcnow().isoformat() + "Z",
        "data": data
    }

    # Untuk log di server
    print("📥 Parking status received:",
          latest_parking_status["received_at"],
          "| slots:", data.get("total_slots"))

    return jsonify({"message": "status received"}), 200


@app.route("/api/parking-status", methods=["GET"])
def get_parking_status():
    if latest_parking_status is None:
        return jsonify({"message": "no data yet"}), 404
    return jsonify(latest_parking_status), 200


if __name__ == "__main__":
    # Jalankan server di port 8000 (sesuai dengan API_URL di client)
    app.run(host="0.0.0.0", port=8000, debug=True)
