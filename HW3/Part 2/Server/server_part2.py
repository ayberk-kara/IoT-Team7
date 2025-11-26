# server_part2.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

# ================================
# Pydantic Model Definition
# ================================

class CountData(BaseModel):
    count: int
    timestamp: float

# ================================
# Shared Variables
# ================================

latest_count = 0
count_lock = asyncio.Lock()

latency_records = []
latency_lock = asyncio.Lock()

# ================================
# API Endpoints
# ================================

@app.post("/update_count")
async def update_count(data: CountData):
    """
    Receives the people count and timestamp from the device.
    Calculates latency and updates the latest count.
    """
    global latest_count
    current_time = datetime.utcnow().timestamp()
    latency = (current_time - data.timestamp) * 1000  # Convert to milliseconds

    # Update the latest count with thread safety
    async with count_lock:
        latest_count = data.count

    # Record latency with thread safety
    async with latency_lock:
        latency_records.append(latency)

    print(f"[INFO] Received count: {data.count}, Latency: {latency:.2f} ms")

    return {"status": "success", "latency_ms": latency}

@app.get("/people_count")
async def get_people_count():
    """
    Streams the latest number of people detected.
    """
    async with count_lock:
        return {"people_count": latest_count}

@app.get("/average_latency")
async def get_average_latency():
    """
    Provides the average latency of processing.
    """
    async with latency_lock:
        if not latency_records:
            avg_latency = 0
        else:
            avg_latency = sum(latency_records) / len(latency_records)
    return {"average_latency_ms": avg_latency}

# ================================
# Server Runner
# ================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
