import requests
import time
from threading import Thread

ip = input("Input the IP:")
address = f"http://{ip}:80"

def listen_to_get_endpoint():
    while True:
        try:
            response = requests.get(f"{address}/people_count")
            if response.status_code == 200:
                print(response.json())
            else:
                print(f"Failed to get data, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error in GET request: {e}")
       
        time.sleep(0.1)

get_thread = Thread(target=listen_to_get_endpoint, daemon=True)
get_thread.start()

while True:
    time.sleep(0.01)