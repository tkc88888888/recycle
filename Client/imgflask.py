import datetime

import requests

t1 = datetime.datetime.now()
image1 = open("image1top.jpg", "rb")
image2 = open("image1front.jpg", "rb")

data = {"weightattr": 55}

file_list = [
    ('image1top', ('image1top.jpg', open('image1top.jpg', 'rb'), 'image/jpg')),
    ('image1front', ('image1front.jpg', open('image1front.jpg', 'rb'), 'image/jpg'))
]

try:
    # response = requests.post('http://localhost:8000/classify_image', files=file_list, data=data)
    response = requests.post('http://wastesorting.dlinkddns.com:8000/classify_image', files=file_list, data=data)
    # response = requests.post('http://wastesorting.dlinkddns.com:8000/classify_image', data=data)
    print(response.text)
    print(' time taken: {} '.format(datetime.datetime.now() - t1))
    dict = response.text

except requests.exceptions.RequestException as e:
    print("Error sending reading: {}".format(e))
