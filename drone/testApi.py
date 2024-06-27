import requests

def test_ipify():
    try:
        response = requests.get('https://api64.ipify.org?format=json', timeout=5)
        if response.status_code == 200:
            print("Successfully fetched IP address:", response.json()['ip'])
        else:
            print("Failed to fetch IP address:", response.status_code)
    except Exception as e:
        print("Error fetching IP address:", str(e))

if __name__ == "__main__":
    test_ipify()
