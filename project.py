from ultralytics import YOLO
import cv2
import cvzone
import math
from pydub import AudioSegment
from pydub.playback import play

# Load the audio file
alert_sound = AudioSegment.from_mp3("synthesize.mp3")

cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video

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

    no_safety_gear_detected = False  # Flag to track if any safety gear is missing

    results = model(img, stream=True)
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
                    no_safety_gear_detected = True  # Set flag if any safety gear is missing

                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    if no_safety_gear_detected:
        play(alert_sound)  # Play the alert sound if any safety gear is missing

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
