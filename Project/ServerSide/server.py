import os
import json
import time
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import csv
import logging
import classifier
import cache_handler
import helpers
import constants
import auth

if not os.path.exists(constants.CACHE_DIR):
    os.makedirs(constants.CACHE_DIR)
if not os.path.exists(constants.PEOPLE_DIR):
    os.makedirs(constants.PEOPLE_DIR)

# -----------------------------
# Global Variables
# -----------------------------
file_counter = {}
verify_fall = {}
barometer_points = {}

# -----------------------------
# Initialize Flask and SocketIO
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all origins
socketio = SocketIO(app, cors_allowed_origins="*")
cache_handler.clean()

# -----------------------------
# Serve the Client HTML
# -----------------------------
@app.route('/')
def index():
    return render_template('client.html')

# -----------------------------
# /getinfo: Provide Current Info
# -----------------------------
@app.route('/getinfo', methods=['GET'])
def get_info():
    if auth.check(request.headers.get('passkey')) == False:
        return jsonify({"message": "Invalid passkey"}), 403
    people_info = {}
    if constants.PEOPLE_DIR.exists() and constants.PEOPLE_DIR.is_dir():
        for person_dir in constants.PEOPLE_DIR.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name.replace('_', ' ')
                latest_file = sorted(person_dir.glob('*.json'), key=os.path.getmtime, reverse=True)
                if not latest_file:
                    people_info[person_name] = {
                        'latitude': None,
                        'longitude': None,
                        'battery_level': None,
                        'floor': None,
                        'help': False,
                    }
                    continue
                latest_file = latest_file[0]
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for file '{latest_file}': {e}")
                    continue

                latitude = None
                longitude = None
                battery_level = None
                floor = "Unknown"
                fall = False

                location_data = helpers.get_element(data, "location")
                if location_data:
                    latitude = location_data.get("values", {}).get("latitude")
                    longitude = location_data.get("values", {}).get("longitude")

                battery_data = helpers.get_element(data, "battery")
                if battery_data:
                    battery_level = battery_data.get("values", {}).get("batteryLevel")
                    if battery_level is not None:
                        battery_level = round(battery_level * 100)

                barometer_data = helpers.get_element(data, "barometer")
                if barometer_data:
                    relativeAltitude = barometer_data.get("values", {}).get("relativeAltitude")
                    floor = helpers.determine_floor(relativeAltitude)

                fall_data = helpers.get_element(data, "fall")
                if fall_data:
                    fall = bool(int(fall_data.get("values", {}).get("detected")))

                people_info[person_name] = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'battery_level': battery_level,
                    'floor': floor,
                    'help': fall,
                }

    return jsonify(people_info), 200

# -----------------------------
# /auth: Authenticate User
# -----------------------------
@app.route('/auth', methods=['POST'])
def handle_auth():
    data = request.get_json() or {}
    
    passkey = data.get('passkey')
    if not passkey:
        return jsonify({"message": "Missing passkey"}), 400
    if auth.check(passkey):
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid passkey"}), 403

# -----------------------------
# Execute model to get prediction result
# -----------------------------
def run_prediction(person):
    csv_file = cache_handler.merge(person)
    prediction = classifier.predict(csv_file)
    os.remove(csv_file)

    return prediction

# -----------------------------
# Verify fall in 2 steps
# -----------------------------
def verify(person):

    csv_file = cache_handler.merge_latest_N(person, 6)

    # 1. Check flatlining
    flat = classifier.check_flatline(csv_file)

    # 2. Check altitude change
    min_height = min(barometer_points[person])
    max_height = max(barometer_points[person])
    diff = max_height - min_height

    # Determine fall
    fall = flat and (diff > constants.FALL_HEIGHT_THRESHOLD)

    min_index = barometer_points[person].index(min_height)
    max_index = barometer_points[person].index(max_height)

    os.remove(csv_file)

    # Represents upwards motion
    if min_index < max_index:
        print(f"Verifying failed for '{person}'")
        return False

    print(f"Verification result for '{person}': {fall}")
    return fall

# -----------------------------
# /stream: Endpoint for streaming from device
# -----------------------------
@app.route('/stream', methods=['POST'])
def handle_post():

    global file_counter, verify_fall, barometer_points
    fall = 0

    # Parse person name from authorization header
    person = request.headers.get('Authorization')
    if not person:
        return jsonify({'message': 'Authorization header missing.'}), 400

    # Initialization
    if person not in file_counter:
        print(person,"connected!")
        file_counter[person] = constants.FRAME_SIZE

    if person not in verify_fall:
        verify_fall[person] = []

    if person not in barometer_points:
        barometer_points[person] = []

    # JSON data from message
    payload_data = request.get_json().get("payload", {})

    file_name = f"{constants.CACHE_DIR}/{person}_{int(time.time() * 100)}.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for entry in payload_data:
            if entry['name'] == 'accelerometer':
                values = entry['values']
                writer.writerow([entry['time'], values.get('x', ''), values.get('y', ''), values.get('z', '')])
    file_counter[person] -= 1
    
    # Enough data for prediction
    if file_counter[person] == 0:
        
        # Decrement all numbers in array by 1
        verify_fall[person] = [x - 1 for x in verify_fall[person]]
        
        if 0 in verify_fall[person]:
            verify_fall[person].remove(0)
            fall = int(verify(person))

        cache_handler.delete_oldest_N(person, 3)

        fall_detected = run_prediction(person)
        if fall_detected == 1:
           verify_fall[person].append(2)
        print(person, ">> Model prediction:", fall_detected)
        
        file_counter[person] = 3

    # Extract location/battery
    latitude = None
    longitude = None
    battery_level = None
    relativeAltitude = None

    location_data = helpers.get_element(payload_data, "location")
    if location_data:
        latitude = location_data.get("values", {}).get("latitude")
        longitude = location_data.get("values", {}).get("longitude")

    battery_data = helpers.get_element(payload_data, "battery")
    if battery_data:
        battery_level = battery_data.get("values", {}).get("batteryLevel")
    
    barometer_data = helpers.get_element(payload_data, "barometer")
    if barometer_data:
        relativeAltitude = barometer_data.get("values", {}).get("relativeAltitude")
        barometer_points[person].append(relativeAltitude)
        if len(barometer_points[person]) > 13:
            del barometer_points[person][0]

    sanitized_person = person.replace(' ', '_')
    person_dir = constants.PEOPLE_DIR / sanitized_person
    person_dir.mkdir(parents=True, exist_ok=True)

    # Load existing data if the file exists
    filename = person_dir / f"{sanitized_person}.json"
    if filename.exists():
        with open(filename, "r") as f:
            current_data = json.load(f)
    else:
        current_data = []

    # Update or add the location
    if latitude is not None and longitude is not None:
        location_entry = next((item for item in current_data if item["name"] == "location"), None)
        if location_entry:
            location_entry["values"] = {"latitude": latitude, "longitude": longitude}
        else:
            current_data.append({
                "name": "location",
                "values": {"latitude": latitude, "longitude": longitude}
            })

    # Update or add the battery
    if battery_level is not None:
        battery_entry = next((item for item in current_data if item["name"] == "battery"), None)
        if battery_entry:
            battery_entry["values"] = {"batteryLevel": battery_level}
        else:
            current_data.append({
                "name": "battery",
                "values": {"batteryLevel": battery_level}
            })
    
    # Update or add the battery
    if relativeAltitude is not None:
        barometer_entry = next((item for item in current_data if item["name"] == "barometer"), None)
        if barometer_entry:
            barometer_entry["values"] = {"relativeAltitude": relativeAltitude}
        else:
            current_data.append({
                "name": "barometer",
                "values": {"relativeAltitude": relativeAltitude}
            })
    
    # Update or add the fall
    fall_entry = next((item for item in current_data if item["name"] == "fall"), None)
    if fall_entry:
        if fall_entry["values"]["detected"] == "0":
            fall_entry["values"] = {"detected": str(fall)}
    else:
        current_data.append({
            "name": "fall",
            "values": {"detected": str(fall)}
        })

    # Write the updated data back to the file
    with open(filename, "w") as f:
        json.dump(current_data, f, indent=4)   

    return "Message received", 200

# -----------------------------
# /delete/<person>: Remove Person Data
# -----------------------------
@app.route('/delete/<string:person>', methods=['DELETE'])
def delete_person(person):
    passkey = request.headers.get('passkey')

    if not passkey:
        abort(400, description="Missing Name or Token in headers.")
    if not auth.check(passkey):
        abort(403, description="Unauthorized.")

    sanitized_person = person.replace(' ', '_')
    person_dir = constants.PEOPLE_DIR / sanitized_person

    if person_dir.exists() and person_dir.is_dir():
        try:
            for file in person_dir.glob('*'):
                file.unlink()
            person_dir.rmdir()
        except OSError as e:
            return jsonify({'message': 'Failed to delete person data.'}), 500

        # Emit a real-time event to notify clients
        socketio.emit('person_deleted', {'person': person})

        return jsonify({'message': f'{person} has been deleted.'}), 200
    else:
        return jsonify({'message': 'Person not found.'}), 404

# -----------------------------
# Socket.IO Events
# -----------------------------
@socketio.on('connect')
def handle_connect():
    emit('connection_response', {'message': 'Connected to server'})

# -----------------------------
# Run the Server
# -----------------------------
if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)