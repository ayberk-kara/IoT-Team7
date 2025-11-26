import os
import shutil
import constants
import uuid
import csv
from glob import glob

# -----------------------------
# Used for handling CSVs in cache
# Each CSV file corresponds to 1 second of data
# -----------------------------
def clean(person=None):
    try:
        for filename in os.listdir(constants.CACHE_DIR):
            file_path = os.path.join(constants.CACHE_DIR, filename)
            if person is None or filename.startswith(person):
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_oldest_N(person, num):
    try:
        csv_files = [
            filename for filename in os.listdir(constants.CACHE_DIR)
            if filename.startswith(person) and filename.endswith(".csv")
        ]
        if len(csv_files) == constants.FRAME_SIZE:
            return
        csv_files.sort(key=lambda f: int(f.rstrip(".csv").split('_')[-1]))
        for i, filename in enumerate(csv_files[:num]):
            file_path = os.path.join(constants.CACHE_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def merge(person):
    file = f"{constants.CACHE_DIR}/{uuid.uuid4()}.csv"
    with open(file, 'w', newline='') as merged_file:
        writer = csv.writer(merged_file)
        writer.writerow(['time', 'x', 'y', 'z'])
        rows_written = 0
        cache_files = glob(f"{constants.CACHE_DIR}/{person}_*.csv")
        cache_files.sort()
        for cache_file_name in cache_files:
            with open(cache_file_name, 'r') as temp_file:
                reader = csv.reader(temp_file)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
        while rows_written < (constants.FRAME_SIZE * constants.SAMPLING_RATE):
            writer.writerow(row)
            rows_written += 1
    return file

def merge_latest_N(person, num):
    file = f"{constants.CACHE_DIR}/{uuid.uuid4()}.csv"
    with open(file, 'w', newline='') as merged_file:
        writer = csv.writer(merged_file)
        writer.writerow(['time', 'x', 'y', 'z'])
        rows_written = 0
        cache_files = glob(f"{constants.CACHE_DIR}/{person}_*.csv")
        cache_files.sort()
        recent_files = cache_files[-num:]
        for cache_file_name in recent_files:
            with open(cache_file_name, 'r') as temp_file:
                reader = csv.reader(temp_file)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
        while rows_written < (constants.FRAME_SIZE * constants.SAMPLING_RATE):
            writer.writerow(row)
            rows_written += 1
    return file