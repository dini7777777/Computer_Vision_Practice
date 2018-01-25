import cv2

#### Loading cascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#### Define a function to do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # drawing rectangle using (img, pt1, pt2, color, thickness)

        cv2.rectangle(img=frame,
                      pt1=(x, y),
                      pt2=(x+w, y+h),
                      color=(255, 0, 0),
                      thickness=2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            # detect eyes inside of face (rio color) to save computation!
            cv2.rectangle(img=roi_color,
                          pt1=(eye_x, eye_y),
                          pt2=(eye_x + eye_w, eye_y + eye_h),
                          color=(0, 255, 0),
                          thickness=2)

    return frame


#### Doing recognition using WebCam
video_capture = cv2.VideoCapture(0)

# keep the cam recognition working
while True:

    # the read() returns two values, we only need the frame
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Doing detection using our function
    canvas = detect(gray, frame)

    # show the results
    cv2.imshow('Video', canvas)

    # Stop the WebCam if we press 'Q' to pause/Quit
    if cv2.waitKey(1) & (0xFF == 'ㄆ' or
                         (0xFF == 'Q' or
                          0xFF == 'q')):
        break

# Turn off WebCam
video_capture.release()

# Close all windows displayed images
cv2.destroyAllWindows()







