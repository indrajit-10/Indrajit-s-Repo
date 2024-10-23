# Create this as face_detection.py
import cv2 #as we need to work with images
import numpy as np 
import os # to detect and find files

class FaceDetector:      # creating a new thing, like face detector, like a robot that can find faces
    def __init__(self):  #tis is the brain of the robot, gives instructions
        
        # Get the path to the cascade file
        cascade_path = os.path.join(cv2.__path__[0], 'data', 
                                  'haarcascade_frontalface_default.xml') # the glasses of the robot to see
                                                                         # find glasses in this file
        self.face_cascade = cv2.CascadeClassifier(cascade_path) # robot given glasses to see faaces
        if self.face_cascade.empty():
            raise ValueError("Failed to load cascade classifier")

    def detect_faces(self, frame):          # giving the glasses picture
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # easier to detect faces in b/w
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(                 # robot is scanning for pictures
            gray,                                                   # checks all parts of the picture
            scaleFactor=1.1,                                        # and looks for areas that look like faces
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:          # loops each face it found
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # draws reatangle box
            cv2.putText(frame, 'Face', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # writes 'face' above the box 
        
        return frame, len(faces)

def main(): # main function
    try:
        # Initialize face detector
        detector = FaceDetector()
        print("Face detector initialized successfully")
        
        # Initialize webcam with DirectShow backend
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # turning on the camera
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # sets camera width =640pixel
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # sets caemra height = 480 pixel
        cap.set(cv2.CAP_PROP_FPS, 30)   # sets fps
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check if it's connected and not in use by another application.")
        
        print("\nFace detection started!")
        print("Press 'q' to quit")
        print("Press 's' to save a snapshot")
        
        while True:                             #robot kepps looking
            # Read frame from webcam
            ret, frame = cap.read()             # robot gets a pic from camera every second
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            frame, face_count = detector.detect_faces(frame)    # uses robot to detect faces in the pic received
            
            # Add face count to frame
            cv2.putText(frame, f'Faces detected: {face_count}', # writes how many faces found on picture
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
            
            # Display the frame
            cv2.imshow('Face Detection', frame) #shows pic with blue box
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF #waiting for a key to be pressed
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                filename = f"face_detection_snapshot_{len(os.listdir('.'))}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved snapshot: {filename}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally: # this part of code will work even if there is a error in the code
        if 'cap' in locals(): # camera is released
            cap.release() 
        cv2.destroyAllWindows() # all display windows are closed propoerly

if __name__ == "__main__":
    main()