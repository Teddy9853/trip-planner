import requests

url = "http://127.0.0.1:8000/plan-trip"

data = {
    "origin": "New York",
    "destination": "Japan",
    "travelers": 2,
    "budget_usd": 3000,
    "days": 5,
    "interests": ["food", "anime", "culture"]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()

    print(result["plan"])

    print("\nMAP STOPS:")
    for stop in result["stops"]:
        print(stop)
else:
    print("Error:", response.status_code)
    print(response.text)