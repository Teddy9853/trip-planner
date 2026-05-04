import requests

url = "http://127.0.0.1:8000/plan-trip"

data = {
    "origin": "New York",
    "destination": "Japan",
    "travelers": 2,
    # Optional:
    "budget_usd": 2000,
    "days": 5,
    "interests": ["food", "anime"]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()

    print("\nFROM:", result["origin"])
    print("TO:", result["destination"])
    print("TRAVELERS:", result["travelers"])
    print("\nPLAN:\n")
    print(result["plan"])
else:
    print("Error:", response.status_code)
    print(response.text)