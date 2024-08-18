from ultralytics import YOLO
import cv2
import cvzone
import math
from playsound import playsound

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Get screen dimensions
screen_width = 1280  # Adjust as needed
screen_height = 720  # Adjust as needed

# Sound flag
#sound_played = False

# Function to process each frame
def process_frame(frame):
   # global sound_played

    # Resize frame to fit within screen dimensions
    img = cv2.resize(frame, (screen_width, screen_height))

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
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    myColor = (0, 0, 255)
                    no_safety_gear_detected = True
                elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    if no_safety_gear_detected :
        playsound('C:\\Users\\dyanu\\PycharmProjects310\\pythonProject\\yello-webcam\\synthesize.mp3')
        #sound_played = True

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not process_frame(frame):
            break
    cap.release()
    cv2.destroyAllWindows()

# Process image
def process_image(image_path):
    frame = cv2.imread(image_path)
    process_frame(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not process_frame(frame):
            break
    cap.release()
    cv2.destroyAllWindows()

# Test different input sources
#process_video("../Videos/ppe4.mp4")
process_image("ppe2.jpg")
#process_webcam()
