import cv2 as cv
import mediapipe as mp
from DD_Functions import *
import pyttsx3
import threading
def speech(type):
    speech = pyttsx3.init()
    voices = speech.getProperty('voices')

    speech.setProperty('voice', voices[1].id)
    if type =='Drowsi':
        speech.say('Cảnh báo buồn ngủ: Có vẻ như bạn đang ngủ.. hãy thức dậy')
    else:
        speech.say('Cảnh báo: Bạn có vẻ mệt mỏi, hãy nghỉ ngơi')
    speech.runAndWait()
    speech.stop()

def speaker(type):
    t1 = threading.Thread(target=speech(type))
    t1.start()

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0, 0, 255), thickness=1, circle_radius=1)

STATIC_IMAGE = False
MAX_NO_FACES = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)

LEFT_EYE_ID = [362, 385,387,263,373,380]
RIGHT_EYE_ID = [33,160,158,133,153,144]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces=MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)


capture = cv.VideoCapture(0)

frame_count = lip_frame = 0
min_frame = 10
min_lip_frame = 5


while True:
    result, image = capture.read()

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:
            draw_landmarks(image, outputs, FACE, COLOR_GREEN)

            #Detect Eyes and Calculate Eye Aspect Ratio
            draw_landmarks(image, outputs, LEFT_EYE_ID, COLOR_RED)
            draw_landmarks(image, outputs, RIGHT_EYE_ID, COLOR_RED)

            landmark = outputs.multi_face_landmarks[0]
            height, width = image.shape[:2]
            leftEye = []
            rightEye = []

            for i in range(len(LEFT_EYE_ID)):
                x = int(landmark.landmark[LEFT_EYE_ID[i]].x * width)
                y = int(landmark.landmark[LEFT_EYE_ID[i]].y * height)
                leftEye.append((x, y))
                next_point = i + 1
                if i == 5:
                    next_point = 0
                x2 = int(landmark.landmark[LEFT_EYE_ID[next_point]].x * width)
                y2 = int(landmark.landmark[LEFT_EYE_ID[next_point]].y * height)
                cv.line(image, (x, y), (x2, y2), COLOR_RED, 1)

            for i in range(len(RIGHT_EYE_ID)):
                x = int(landmark.landmark[RIGHT_EYE_ID[i]].x * width)
                y = int(landmark.landmark[RIGHT_EYE_ID[i]].y * height)
                rightEye.append((x, y))
                next_point = i + 1
                if i == 5:
                    next_point = 0
                x2 = int(landmark.landmark[RIGHT_EYE_ID[next_point]].x * width)
                y2 = int(landmark.landmark[RIGHT_EYE_ID[next_point]].y * height)
                cv.line(image, (x, y), (x2, y2), COLOR_RED, 1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)
            if EAR < 0.225:
                frame_count += 1
            else:
                frame_count = 0
            cv.putText(image, f'EyeFrameC: {int(frame_count)}', (20, 469), cv.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 2)
            if frame_count > min_frame:
                # Closing the eyes
                cv.putText(image, 'Canh bao !!', (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 2)
                speaker('Drowsi')
                frame_count = 0

            # Detect lip and calulate ratio lips
            draw_landmarks(image, outputs, UPPER_LOWER_LIPS, COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)
            ratio_lips = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)

            if ratio_lips < 1.8:
                lip_frame += 1
            else:
                lip_frame = 0
            cv.putText(image, f'LipFrameC: {int(lip_frame)}', (450, 469), cv.FONT_HERSHEY_PLAIN,
                       1, (0, 255, 0), 2)
            if lip_frame > 8:
                #Open his mouth
                cv.putText(image, 'Canh bao !!', (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 2)
                speaker(0)
                lip_frame = 0

        cv.imshow("Drowsiness Detection", image)
        if cv.waitKey(30) & 255 == 27:
            break

capture.release()
cv.destroyAllWindows()