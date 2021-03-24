import requests

resp = requests.post(
    'http://127.0.0.1:5000/predict',
    files={
        "file": open('image_1_68.png', 'rb')
    }
)

print(resp.json())