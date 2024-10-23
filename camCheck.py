import cv2

# Try index 0 first, then other indices like 1, 2, etc.
cap = cv2.VideoCapture(0)  # or try cv2.VideoCapture(1) if you have more cameras

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")
