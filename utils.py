import base64


def decode_image(image_data, image_file):
    image = base64.b64decode(image_data)

    with open(image_file, 'wb') as img:
        img.write(image)
        img.close()
