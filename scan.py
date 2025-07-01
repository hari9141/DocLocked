import cv2
import numpy as np
import os
import time

def scan_document():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Create output directory if it doesn't exist
    output_dir = "scanned_documents"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Press 's' to scan the document, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Display live feed
        cv2.imshow('Document Scanner', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Scan document
            scanned = process_frame(frame)
            if scanned is not None:
                timestamp = int(time.time())
                filename = f"{output_dir}/document_{timestamp}.jpg"
                cv2.imwrite(filename, scanned)
                print(f"Document saved as {filename}")
                cv2.imshow('Scanned Document', scanned)
            else:
                print("No document detected. Try again.")
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(gray, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find document contour
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            screen_cnt = approx
            break
    
    if screen_cnt is None:
        return None
    
    # Apply perspective transform
    warped = four_point_transform(frame, screen_cnt.reshape(4, 2))
    
    # Convert to grayscale and threshold
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(warped_gray, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove shadows (optional)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    mask = thresholded.copy()
    result = cv2.bitwise_and(rgb, rgb, mask=mask)
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def four_point_transform(image, pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB
