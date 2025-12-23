import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# 1. INITIALIZATION & CONSTANTS
# ---------------------------------------------------------

# Webcam Settings
WCAM, HCAM = 640, 480  # Webcam resolution
CAP = cv2.VideoCapture(0)
CAP.set(3, WCAM)
CAP.set(4, HCAM)

# Screen Settings
try:
    WSCR, HSCR = pyautogui.size() # Get screen resolution
except Exception as e:
    print(f"Error getting screen size: {e}")
    print("Defaulting to 1920x1080 for calculation purposes.")
    WSCR, HSCR = 1920, 1080

# Control Parameters
FRAME_R = 100            # Frame Reduction (padding) to reach edges of screen easily
SMOOTHING = 5            # Smoothing factor (higher = smoother but more lag)
CLICK_THRESHOLD = 40     # Distance threshold for clicking
SCROLL_THRESHOLD = 40    # Distance threshold for activating scroll mode
SCROLL_SENSITIVITY = 20  # Pixels per scroll step

# PyAutoGUI Safety
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0        # Disable default pause for faster processing

# Colors (BGR)
COLOR_MOUSE = (255, 0, 255)  # Magenta
COLOR_CLICK = (0, 255, 0)    # Green
COLOR_SCROLL = (0, 255, 255) # Yellow
COLOR_TEXT = (255, 0, 0)     # Blue

# Global State Variables
plocX, plocY = 0, 0      # Previous Location
clocX, clocY = 0, 0      # Current Location
last_click_time = 0      # For debouncing clicks
CLICK_DELAY = 0.3        # Seconds between clicks

# ---------------------------------------------------------
# 2. MEDIAPIPE SETUP (TASKS API)
# ---------------------------------------------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.7,
                                       min_hand_presence_confidence=0.7,
                                       min_tracking_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# Drawing utilities (Using legacy if available, or custom)
# The Tasks API doesn't provide a direct `draw_landmarks` equivalent that works on cv2 images easily out of the box
# in the same way `mp.solutions.drawing_utils` did. We'll implement a simple one or try to import drawing_utils if possible.
try:
    from mediapipe.python.solutions import drawing_utils as mp_draw
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    mp_draw = None

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

# Global Constants for Drawing
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Middle
    (9,13), (13,14), (14,15), (15,16),# Ring
    (13,17), (17,18), (18,19), (19,20),# Pinky
    (0,17)                            # Palm base
]

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two landmarks (x, y)."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    return length, [x1, y1, x2, y2]

def draw_custom_landmarks(img, landmarks):
    """Draws landmarks manually if mp_draw is unavailable."""
    h, w, c = img.shape
    
    # Convert normalized to pixel
    px_points = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        px_points.append((cx, cy))
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(px_points) and p2_idx < len(px_points):
            cv2.line(img, px_points[p1_idx], px_points[p2_idx], (0, 255, 0), 2)

def draw_info(img, fps, mode, gesture_text):
    """Draws UI elements: FPS, current mode, and active gesture."""
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, f'Mode: {mode}', (20, 90), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(img, f'Gesture: {gesture_text}', (20, 130), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Draw Boundary Box for Mouse Movement Range
    cv2.rectangle(img, (FRAME_R, FRAME_R), (WCAM - FRAME_R, HCAM - FRAME_R),
                  (255, 0, 255), 2)


# ---------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------

def main():
    global plocX, plocY, clocX, clocY, last_click_time
    
    pTime = 0
    
    if not CAP.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    print("Virtual Mouse Started.")
    print(" - Right Hand (Index): Move Cursor")
    print(" - Right Hand (Thumb+Index): Left Click")
    print(" - Right Hand (Thumb+Middle): Right Click")
    print(" - Left Hand (Thumb+Index): Pause Cursor (Clutch)")
    print(" - Press 'q' to exit.")

    while True:
        # 1. Capture Frame
        success, img = CAP.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Flip image for mirror view (easier interaction)
        img = cv2.flip(img, 1)
        
        # 2. Process Hand Landmarks
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # Detect
        detection_result = detector.detect(mp_image)
        
        # Current status texts
        mode_status = "Waiting..."
        gesture_status = "None"
        
        cursor_hand = None
        pause_hand = None

        if detection_result.hand_landmarks:
            for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
                
                # Draw landmarks
                draw_custom_landmarks(img, hand_landmarks)
                
                # Identify Hand
                # Note: Because we flipped the image, MP sees the physical "Right" hand as "Left".
                # So: hand_label == "Left" -> Physical Right Hand (Cursor & Click)
                #     hand_label == "Right" -> Physical Left Hand (Pause Clutch)
                hand_label = handedness[0].category_name
                
                # Extract landmark positions for THIS hand
                lmList = []
                for id, lm in enumerate(hand_landmarks):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                
                if hand_label == "Left":
                    cursor_hand = (hand_landmarks, lmList)
                elif hand_label == "Right":
                    pause_hand = (hand_landmarks, lmList)

        # -------------------------------
        # 1. CHECK PAUSE (Physical Left Hand / Label "Right")
        # -------------------------------
        is_paused = False
        if pause_hand:
            _, lmList = pause_hand
            x_thumb, y_thumb = lmList[4][1:]
            x_index, y_index = lmList[8][1:]
            
            # Pinch to Pause (Thumb + Index on Left Hand)
            dist_pause, _ = calculate_distance((x_thumb, y_thumb), (x_index, y_index))
            if dist_pause < CLICK_THRESHOLD:
                is_paused = True
                gesture_status = "PAUSED (LH)"
                cv2.circle(img, (x_thumb, y_thumb), 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x_index, y_index), 15, (0, 0, 255), cv2.FILLED)

        # -------------------------------
        # 2. CURSOR & CLICK (Physical Right Hand / Label "Left")
        # -------------------------------
        if cursor_hand:
            mode_status = "CURSOR (RH)"
            _, lmList = cursor_hand
            
            # Key Landmarks
            x_thumb, y_thumb = lmList[4][1:]
            x_index, y_index = lmList[8][1:]
            x_middle, y_middle = lmList[12][1:]

            # A. Move Cursor (If not paused)
            if not is_paused:
                # Coordinate Conversion & Smoothing
                x3 = np.interp(x_index, (FRAME_R, WCAM - FRAME_R), (0, WSCR))
                y3 = np.interp(y_index, (FRAME_R, HCAM - FRAME_R), (0, HSCR))
                
                # Smoothening
                clocX = plocX + (x3 - plocX) / SMOOTHING
                clocY = plocY + (y3 - plocY) / SMOOTHING
                
                # Move Mouse
                try:
                    pyautogui.moveTo(clocX, clocY)
                except pyautogui.FailSafeException:
                    pass
                    
                plocX, plocY = clocX, clocY
                cv2.circle(img, (x_index, y_index), 15, COLOR_MOUSE, cv2.FILLED) # Visual for cursor
            else:
                mode_status = "PAUSED"
                cv2.circle(img, (x_index, y_index), 15, (0, 0, 255), cv2.FILLED) # Red visual for paused cursor

            # B. Left Click (Thumb + Index)
            dist_lclick, info_lclick = calculate_distance((x_thumb, y_thumb), (x_index, y_index))
            if dist_lclick < CLICK_THRESHOLD:
                cv2.circle(img, (info_lclick[0], info_lclick[1]), 15, COLOR_CLICK, cv2.FILLED)
                cv2.circle(img, (info_lclick[2], info_lclick[3]), 15, COLOR_CLICK, cv2.FILLED)
                
                if time.time() - last_click_time > CLICK_DELAY:
                    pyautogui.click()
                    last_click_time = time.time()
                    gesture_status = "Left Click (RH)"
                    
            # C. Right Click (Thumb + Middle)
            dist_rclick, info_rclick = calculate_distance((x_thumb, y_thumb), (x_middle, y_middle))
            if dist_rclick < CLICK_THRESHOLD:
                cv2.circle(img, (info_rclick[0], info_rclick[1]), 15, COLOR_CLICK, cv2.FILLED)
                cv2.circle(img, (info_rclick[2], info_rclick[3]), 15, COLOR_CLICK, cv2.FILLED)
                
                if time.time() - last_click_time > CLICK_DELAY:
                    pyautogui.rightClick()
                    last_click_time = time.time()
                    gesture_status = "Right Click (RH)"

        # 5. Frame Rate & Display
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        draw_info(img, fps, mode_status, gesture_status)
        
        cv2.imshow("Virtual Mouse - AI Controller", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    CAP.release()
    cv2.destroyAllWindows()
    print("Program exited safely.")

if __name__ == "__main__":
    main()