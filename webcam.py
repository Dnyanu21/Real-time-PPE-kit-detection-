from ultralytics import YOLO
import cv2
import cvzone
import math
from playsound import playsound

#cap = cv2.VideoCapture(0)  # For Webcam
#cap.set(3, 1280)
#cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/ppe4.mp4")  # For Video

#sound_played=False

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

# Get screen dimensions
screen_width = 1280  # Adjust as needed
screen_height = 720  # Adjust as needed

while True:
    success, img = cap.read()
    if not success:
        break

    # Resize frame to fit within screen dimensions
    img = cv2.resize(img, (screen_width, screen_height))

    results = model(img, stream=True)
    no_safety_gear_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                    no_safety_gear_detected = True
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    if no_safety_gear_detected and not sound_played:
        playsound('C:\\Users\\dyanu\\PycharmProjects310\\pythonProject\\yello-webcam\\synthesize.mp3')
        #sound_played=True

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
