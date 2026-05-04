import requests

url = "http://127.0.0.1:8000/plan-trip"

data = {
    "destination": "Japan",
    "budget_usd": 2000,
    "days": 5,
    "interests": ["food", "anime", "culture"]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("\nDESTINATION:", result["destination"])
    print("\nPLAN:\n")
    print(result["plan"])
else:
    print("Error:", response.status_code)
    print(response.text)