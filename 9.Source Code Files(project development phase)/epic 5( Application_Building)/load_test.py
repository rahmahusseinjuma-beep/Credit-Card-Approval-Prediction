import time
import statistics
import threading
import psutil
import requests

URL = "http://127.0.0.1:5000/predict"

PAYLOAD = {
    "gender": "M", "age": "35", "own_car": "Y", "own_realty": "Y",
    "children": "0", "family_members": "2", "income": "250000",
    "income_type": "Working", "education_type": "Higher education",
    "family_status": "Married", "housing_type": "House / apartment",
    "occupation_type": "Managers", "employed_years": "5",
    "work_phone": "Y", "phone": "Y", "email": "Y",
}

response_times = []
errors = 0
lock = threading.Lock()

def send_request():
    global errors
    start = time.time()
    try:
        r = requests.post(URL, data=PAYLOAD, timeout=10)
        elapsed = time.time() - start
        with lock:
            if r.status_code == 200:
                response_times.append(elapsed)
            else:
                errors += 1
    except Exception:
        with lock:
            errors += 1

# --- Resource monitoring in the background ---
cpu_samples = []
mem_samples = []
monitoring = True

def monitor():
    import subprocess
    pid_out = subprocess.run(["pgrep", "-f", "python3 app.py"], capture_output=True, text=True).stdout.strip()
    pids = [int(p) for p in pid_out.splitlines() if p]
    if not pids:
        print("monitor: could not find app.py process")
        return
    proc = psutil.Process(pids[0])
    proc.cpu_percent(None)
    samples_taken = 0
    while monitoring and samples_taken < 100:
        time.sleep(0.1)
        try:
            cpu_samples.append(proc.cpu_percent(None))
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))
        except Exception:
            break
        samples_taken += 1

mon_thread = threading.Thread(target=monitor, daemon=True)
mon_thread.start()

# --- Load test: 20 virtual users, each sending 5 requests = 100 total ---
NUM_USERS = 20
REQUESTS_PER_USER = 5

start_time = time.time()
threads = []
for _ in range(NUM_USERS):
    for _ in range(REQUESTS_PER_USER):
        t = threading.Thread(target=send_request)
        threads.append(t)
        t.start()
for t in threads:
    t.join()
total_time = time.time() - start_time

monitoring = False
mon_thread.join()

# --- Results ---
total_requests = NUM_USERS * REQUESTS_PER_USER
success_count = len(response_times)

print("=== Load Test Results ===")
print(f"Total requests sent: {total_requests}")
print(f"Successful responses: {success_count}")
print(f"Errors: {errors}")
print(f"Total wall time: {total_time:.2f} sec")
print(f"Throughput: {success_count / total_time:.2f} req/sec")
print()
if response_times:
    print(f"Avg response time: {statistics.mean(response_times):.3f} sec")
    print(f"Max response time: {max(response_times):.3f} sec")
    print(f"Min response time: {min(response_times):.3f} sec")
    sorted_times = sorted(response_times)
    p95_idx = int(len(sorted_times) * 0.95)
    print(f"95th percentile: {sorted_times[min(p95_idx, len(sorted_times)-1)]:.3f} sec")
print()
error_rate = (errors / total_requests) * 100
print(f"Error rate: {error_rate:.2f}%")
print()
if cpu_samples:
    print(f"Avg CPU utilization (Flask process): {statistics.mean(cpu_samples):.1f}%")
    print(f"Max CPU utilization (Flask process): {max(cpu_samples):.1f}%")
if mem_samples:
    print(f"Avg memory usage (Flask process): {statistics.mean(mem_samples):.1f} MB")
    print(f"Max memory usage (Flask process): {max(mem_samples):.1f} MB")
