# test_wsl_connection.py
import requests
import sys

# Use the SAME URL as in your main config
# server_url = "http://localhost:5001"
server_url = "http://127.0.0.1:5001"
status_endpoint = "/status"
full_url = f"{server_url.rstrip('/')}/{status_endpoint.lstrip('/')}"

print(f"Attempting to connect to: {full_url}")

try:
    response = requests.get(full_url, timeout=5.0) # Use a short timeout
    response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
    data = response.json()
    print("SUCCESS!")
    print(f"Status: {response.status_code}")
    print(f"Response JSON: {data}")
    sys.exit(0) # Exit with success code

except requests.exceptions.ConnectionError as e:
    print(f"\n!!! CONNECTION ERROR: {e}")
    print("This indicates a network-level issue reaching the server (like connection refused).")
except requests.exceptions.Timeout as e:
    print(f"\n!!! TIMEOUT ERROR: {e}")
    print("The server was reached but didn't respond in time.")
except requests.exceptions.HTTPError as e:
    print(f"\n!!! HTTP ERROR: {e}")
    print(f"Server responded with error status: {e.response.status_code}")
    print(f"Response text: {e.response.text[:200]}...")
except requests.exceptions.RequestException as e:
    print(f"\n!!! REQUESTS LIBRARY ERROR: {e}")
except Exception as e:
    print(f"\n!!! UNEXPECTED PYTHON ERROR: {e}")

sys.exit(1) # Exit with error code