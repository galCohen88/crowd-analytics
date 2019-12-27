import math

import boto3 as boto3
import cv2
import numpy as np


access_key = ''
secrete_key = ''
brightness = 0.9
camera = 0
image_name = 'tmp.jpg'


def change_brightness(img, alpha, beta):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 1, beta)


def take_picture(img_name):
    cam = cv2.VideoCapture(camera)
    ret, frame = cam.read()
    frame = change_brightness(frame, brightness, 0)

    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    cam.release()

font_emotions = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
font_emotions_scale = 0.3
font_emotions_color = (0, 128, 255)
lineType = 1


while True:
    total_calm = 0
    total_confused = 0
    total_happy = 0

    take_picture(image_name)
    with open(image_name, 'rb') as f: # open the file in read binary mode
        data = f.read()

    client = boto3.client('rekognition', aws_access_key_id=access_key, aws_secret_access_key=secrete_key)

    print('detecting')

    response = client.detect_faces(
        Image={
            'Bytes': data
        },
        Attributes=['ALL']
    )
    img = cv2.imread(image_name)

    male = 0
    female = 0

    for face in response.get('FaceDetails'):
        width = face.get('BoundingBox').get('Width')
        height = face.get('BoundingBox').get('Height')
        left = face.get('BoundingBox').get('Left')
        top = face.get('BoundingBox').get('Top')
        gender = face.get('Gender').get('Value')

        emotions_dict = {}
        for item in face.get('Emotions'):
            type = item['Type']
            emotions_dict[type] = item['Confidence']

        calm = str(int(emotions_dict.get('CALM', 0)))
        confused = str(int(emotions_dict.get('CONFUSED', 0)))
        happy = str(int(emotions_dict.get('HAPPY', 0)))

        im_height, im_width, channels = img.shape

        face_width = int(width * im_width)
        face_height = int(height * im_height)
        face_left_position = int(left * im_width)
        face_top_position = int(top * im_height)

        if gender == 'Male':
            male += 1
            color = (250, 206, 135)
        else:
            female +=1
            color = (51, 51, 255)
        img = cv2.rectangle(img, (face_left_position, face_top_position), (face_left_position+face_width, face_height+face_top_position), color, 2)

        total_calm += int(calm)
        total_confused += int(confused)
        total_happy += int(happy)

        # calm = f'{calm}%'
        # surprised = f'{surprised}%'
        # confused = f'{confused}%'
        # bottomLeftCornerOfText = (face_left_position+face_width+10, face_top_position + 40)
        # cv2.putText(img, confused, bottomLeftCornerOfText, font_emotions, font_emotions_scale, font_emotions_color, lineType)
        # bottomLeftCornerOfText = (face_left_position+face_width+10, face_top_position + 30)
        # cv2.putText(img, surprised, bottomLeftCornerOfText, font_emotions, font_emotions_scale, font_emotions_color, lineType)
        # bottomLeftCornerOfText = (face_left_position+face_width+10, face_top_position + 20)
        # cv2.putText(img, calm, bottomLeftCornerOfText, font_emotions, font_emotions_scale, font_emotions_color, lineType)

    if male+female == 0:
        cv2.imshow("window", img)
        cv2.moveWindow('window', 200, 200)
        cv2.waitKey(1)
        continue
    female_percentage = math.ceil((female / float(male+female)) * 100)
    male_percentage = (male / float(male+female)) * 100

    font = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2

    cv2.putText(img, str(int(male_percentage)) + '% Male, ' + str(int(female_percentage))
                + '% Female, A. Happy ' + str(int(total_happy/float(male+female)))
                + '%, A. Calm ' + str(int(total_calm/float(male+female)))
                + '%, A. Confused ' + str(int(total_confused / float(male + female))) + '%',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow("window", img)
    cv2.moveWindow('window', 200, 200)
    cv2.waitKey(1)
