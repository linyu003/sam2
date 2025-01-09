from PIL import Image
import requests

# image = Image.open('images/cars.jpg')

with open('images/cars.jpg','rb') as f:
    res = requests.post('http://localhost:8001/gen_mask',files={'image':f})
print(res.text)
