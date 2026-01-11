import requests
import json
import os

BASE_URL = "http://localhost:8000"

def test_api():
    print("--- Testing Backend API ---")
    
    # 1. Test Home (Tasks list)
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"GET /: {r.status_code}")
        if r.status_code == 200:
            tasks = r.json().get('tasks', [])
            print(f"Found {len(tasks)} tasks.")
        else:
            print("Error: Could not fetch tasks")
            return
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        return

    # 2. Test Task Detail Config
    if tasks:
        task_id = tasks[0]['id']
        r = requests.get(f"{BASE_URL}/tasks/{task_id}/")
        print(f"GET /tasks/{task_id}/: {r.status_code}")
        
    print("\n--- API Test Complete ---")
    print("Backend is ready for tunneling.")

if __name__ == "__main__":
    test_api()
