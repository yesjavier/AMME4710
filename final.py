import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import pyautogui


#HAND GESTURES ARE REALLY GITTERY AND THE MOUSE COUNTROL IS NOT ACCURATE

# Initialize Mediapipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Pygame Setup
pygame.init()
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))  # Double the width for whiteboard + webcam
pygame.display.set_caption("Whiteboard and Webcam")
brush_size = 5

# Create a surface for the whiteboard
whiteboard = pygame.Surface((WIDTH, HEIGHT))
whiteboard.fill(WHITE)

# Webcam capture setup
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

drawing = False
last_pos = None
cursor_pos = (0, 0)  # Initialize cursor position

alpha = 0.6
smoothed_cursor = np.array([0, 0], dtype=float)



# Recognize gestures function (adapted from previous code)
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))



def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_base = landmarks[5]
    index_tip = landmarks[8]
    middle_base = landmarks[9]
    middle_tip = landmarks[12]
    ring_base = landmarks[13]
    ring_tip = landmarks[16]
    pinky_base = landmarks[17]
    pinky_tip = landmarks[20]
    palm_base = landmarks[0]

    # Calculate distances between relevant points (for basic rule-based recognition)
    thumb_index_distance = calculate_distance(thumb_tip[:2], index_tip[:2])
    thumb_palm_distance = calculate_distance(thumb_tip[:2], palm_base[:2])
    index_tipbase_distance = calculate_distance(index_tip[:2], index_base[:2])
    middle_tipbase_distance = calculate_distance(middle_tip[:2], middle_base[:2])
    ring_tipbase_distance = calculate_distance(ring_tip[:2], ring_base[:2])
    pinky_tipbase_distance = calculate_distance(pinky_tip[:2], pinky_base[:2])
    index_palm_distance = calculate_distance(index_tip[:2], palm_base[:2])
    middle_palm_distance = calculate_distance(middle_tip[:2], palm_base[:2])
    ring_palm_distance = calculate_distance(ring_tip[:2], palm_base[:2])
    pinky_palm_distance = calculate_distance(pinky_tip[:2], palm_base[:2])

    # Example gesture recognition rules
    if thumb_index_distance < 60 and middle_tipbase_distance < 80 and ring_tipbase_distance < 80: #WHEN HAND is not parallel to camera, distance between tip and base of fingers is larger

        return "Fist"
    elif index_palm_distance > 150 and middle_palm_distance > 150 and ring_palm_distance > 150 and thumb_palm_distance > 100:
        return "Open Hand"
    else:
        return "Unknown Gesture"

# Main loop
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.8) as hands:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    whiteboard.fill(WHITE)  # Clear the whiteboard

        # Capture frame-by-frame from webcam
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the image with Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Reflect (mirror) the image along the Y-axis (flip horizontally)
        # frame = cv2.flip(frame, 1)





        # Gesture recognition and cursor tracking
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                landmarks = [[lm.x * WIDTH, lm.y * HEIGHT, lm.z] for lm in hand_landmarks.landmark]

                # Recognize gesture
                gesture = recognize_gesture(landmarks)

                # Use the index finger as the "cursor"
                # Flip the x-coordinate of the index finger (invert the horizontal axis)
                wrist = (WIDTH - int(landmarks[0][0]), int(landmarks[0][1]))

                # Update cursor position
                current_pos = np.array(wrist, dtype=float)
                
                

                # Apply exponential smoothing
                smoothed_cursor = alpha * current_pos + (1 - alpha) * smoothed_cursor
                cursor_pos = tuple(smoothed_cursor.astype(int))

                # Cursor movement logic
                
                if gesture == "Fist":
                    # Move the cursor without drawing (no left click)
                    last_pos = cursor_pos
                elif gesture == "Open Hand":
                    # Move the cursor and draw (simulate left click)
                    drawing = True
                    
                    if last_pos is not None:
                        pygame.draw.line(whiteboard, BLACK, last_pos, cursor_pos, brush_size)
                    last_pos = cursor_pos
                else:
                    drawing = False
                    last_pos = None


          
                # Display recognized gesture on the frame

                text_image = np.zeros_like(frame)
                
                cv2.putText(text_image, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                text_image = cv2.flip(text_image, 1)

                frame = cv2.addWeighted(frame, 1, text_image, 1, 0)

                

        # Convert the webcam frame to a pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)  # Rotate to match pygame window orientation
        frame_surface = pygame.surfarray.make_surface(frame)

        # Blit both the webcam feed and whiteboard on the pygame screen
        screen.blit(frame_surface, (WIDTH, 0))  # Display webcam on the right side
        screen.blit(whiteboard, (0, 0))  # Display whiteboard on the left side

        # Draw the red dot to track the cursor
        pygame.draw.circle(screen, RED, cursor_pos, 10)  # Red dot on the whiteboard

        # Update the display
        pygame.display.flip()

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'ESC'
            break

cap.release()
pygame.quit()
