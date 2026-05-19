import time
import requests
import uuid
import sys

BASE_URL = "http://localhost:8000"

def run_performance_test():
    print("="*50)
    print(" VisCurator Performance & Load Testing")
    print("="*50)

    # 1. Measure System Health Latency
    start = time.time()
    res = requests.get(f"{BASE_URL}/api/health")
    health_latency = time.time() - start
    print(f"[Health Check] Latency: {health_latency:.4f} seconds")

    if res.status_code != 200:
        print("Backend is down or unreachable. Exiting performance test.")
        sys.exit(1)

    # 2. Measure Dataset Browse (API Search) Latency
    start = time.time()
    res = requests.get(f"{BASE_URL}/api/dataset/browse", params={"q": "dogs", "max_results": 5})
    browse_latency = time.time() - start
    print(f"[Dataset Search] Latency: {browse_latency:.4f} seconds (Query: 'dogs')")

    # 3. Measure System Telemetry Latency
    start = time.time()
    res = requests.get(f"{BASE_URL}/api/system/telemetry")
    telemetry_latency = time.time() - start
    print(f"[System Telemetry] Latency: {telemetry_latency:.4f} seconds")

    print("\n[Analysis Ready]")
    print(f"Total API interaction time: {health_latency + browse_latency + telemetry_latency:.4f} seconds")
    print("These metrics can be included in the 'Performance Analysis' section of your report.")
    print("For full end-to-end pipeline performance, we recommend measuring the SAM/CLIP annotation time")
    print("using the websocket logs or timestamping the job creation vs completion in the frontend.")
    print("="*50)

if __name__ == "__main__":
    try:
        run_performance_test()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the backend. Is FastAPI running on port 8000?")
