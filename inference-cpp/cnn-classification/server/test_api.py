import requests, json, base64

url = "http://localhost:8181/predict"

image_path = "../geeks.png"

result = requests.post(url, json={"image": base64.b64encode(open(image_path, "rb").read()).decode()}).text

#def png_to_base64(PNGPath:str):
#    import base64
#    with open(PNGPath,"rb") as image_file:
#        encoded_bytes = base64.b64encode(image_file.read())
#        return encoded_bytes
#
#def packet_factory_single_pattern(PNGPath:str):
#    img = png_to_base64(PNGPath)
#    data = {}
#    data['image'] = img.decode('latin1')
#    return data
#
#data=packet_factory_single_pattern(image_path)
#
#response = requests.post(url='http://localhost:8181/predict', json=data)
#
#print(response)
print(json.loads(result))
