import requests
import json

def test_api():
    url = "http://127.0.0.1:8000/predict"
    
    # Example data mirroring the dataset schema
    sample_data = {
        "network_packet_size": 599.0,
        "protocol_type": "TCP",
        "login_attempts": 4,
        "session_duration": 492.98,
        "encryption_used": "DES",
        "ip_reputation_score": 0.6,
        "failed_logins": 1,
        "browser_type": "Edge",
        "unusual_time_access": 0
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=sample_data)
        if response.status_code == 200:
            print("Successfully received response!")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_api()
