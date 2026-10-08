# Allows us to measure how long each frame takes to process.
import time

# Lets us work with file paths reliably.
from pathlib import Path

# OpenCV: opens the webcam, displays video, and draws on frames.
import cv2

# MediaPipe: runs the hand and object-detection models.
import mediapipe as mp


# -------------------- MODEL FILE PATHS --------------------

# The pre-trained EfficientDet-Lite0 object detector.
# Later, replace this with your forceps-trained .tflite model.
OBJECT_MODEL_PATH = Path("models/efficientdet_lite0.tflite")

# MediaPipe model that finds 21 landmarks on each hand.
HAND_MODEL_PATH = Path("models/hand_landmarker.task")


# Stop immediately if either downloaded model file is missing.
if not OBJECT_MODEL_PATH.exists() or not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        "Check that both model files are inside the models folder."
    )


# -------------------- MEDIAPIPE CLASS SHORTCUTS --------------------

# BaseOptions tells MediaPipe where a model file is located.
BaseOptions = mp.tasks.BaseOptions

# VIDEO mode is used because we are processing continuous webcam frames.
RunningMode = mp.tasks.vision.RunningMode

# EfficientDet-related classes.
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

# Hand-tracking-related classes.
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# -------------------- OBJECT DETECTOR SETTINGS --------------------

object_options = ObjectDetectorOptions(
    # Load the EfficientDet-Lite0 model file.
    base_options=BaseOptions(model_asset_path=str(OBJECT_MODEL_PATH)),

    # Tell MediaPipe that input comes from a video/webcam.
    running_mode=RunningMode.VIDEO,

    # Return up to two detected objects per frame.
    # Later, this easily covers two forceps.
    max_results=2,

    # Ignore detections with confidence below 60%.
    score_threshold=0.60,
)


# -------------------- HAND TRACKER SETTINGS --------------------

hand_options = HandLandmarkerOptions(
    # Load the hand-landmark model file.
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),

    # Webcam frames are video input.
    running_mode=RunningMode.VIDEO,

    # Detect up to two hands at the same time.
    num_hands=2,

    # Minimum confidence required to detect a hand.
    min_hand_detection_confidence=0.50,

    # Minimum confidence that a detected hand is still present.
    min_hand_presence_confidence=0.50,

    # Minimum confidence required to keep tracking a hand over time.
    min_tracking_confidence=0.50,
)


# -------------------- HAND SKELETON CONNECTIONS --------------------

# MediaPipe returns 21 hand landmarks numbered 0 through 20.
# Each pair tells us which landmarks should be connected by a line.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Base of the hand
]


# -------------------- OPEN THE WEBCAM --------------------

# 0 means “use the default camera.”
# If you have multiple cameras, try 1 instead.
camera = cv2.VideoCapture(0)

# Stop with a clear error if OpenCV could not access the webcam.
if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# Create both MediaPipe models once.
# They stay open while the webcam loop runs.
with ObjectDetector.create_from_options(object_options) as forceps_detector, \
     HandLandmarker.create_from_options(hand_options) as hand_tracker:

    # Keep reading frames until the user presses q.
    while True:
        # Read one frame from the webcam.
        success, frame = camera.read()

        # End the program if the camera stops returning frames.
        if not success:
            break

        # Start a timer so we can calculate combined FPS later.
        start = time.perf_counter()

        # Get frame dimensions in pixels.
        height, width = frame.shape[:2]

        # OpenCV frames use BGR color order.
        # MediaPipe expects RGB, so convert the frame.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the RGB image in MediaPipe's image format.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # Video mode requires timestamps that always increase.
        timestamp_ms = int(time.monotonic() * 1000)

        # Send the same frame to EfficientDet.
        # Currently it detects common objects.
        # After training, it should detect forceps.
        forceps_result = forceps_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # Send the same frame to the hand tracker.
        hands_result = hand_tracker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # Print landmarks for debugging.
        """
        for hand_idx, hand_landmarks in enumerate(hands_result.hand_landmarks):
            print(f"\nHand {hand_idx} (image coordinates):")

            for i, lm in enumerate(hand_landmarks):
                print(f"  Landmark {i}: x={lm.x:.17f}, y={lm.y:.17f}, z={lm.z:.17f}")

        for hand_idx, world_landmarks in enumerate(hands_result.hand_world_landmarks):
            print(f"\nHand {hand_idx} (world coordinates):")

            for i, lm in enumerate(world_landmarks):
                print(f"  Landmark {i}: x={lm.x:.17f}, y={lm.y:.17f}, z={lm.z:.17f}")  
        """
        # -------------------- DRAW OBJECT BOXES --------------------

        # Go through every object EfficientDet found.
        for detection in forceps_result.detections:
            # Get the detected object's rectangle.
            box = detection.bounding_box

            # Get the highest-confidence label for that object.
            category = detection.categories[0]

            # Bounding box position and size, in pixels.
            x = box.origin_x
            y = box.origin_y
            w = box.width
            h = box.height

            # Draw a green rectangle around the detected object.
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),  # Green in BGR format
                2,            # Line thickness
            )

            # Create text such as "cup: 0.82".
            label = f"{category.category_name}: {category.score:.2f}"

            # Draw the label just above the box.
            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # -------------------- DRAW HAND LANDMARKS --------------------

        # Each item in hand_landmarks represents one detected hand.
        for hand_landmarks in hands_result.hand_landmarks:
            points = []

            # Convert normalized landmark positions to image pixels.
            for landmark in hand_landmarks:
                # landmark.x and landmark.y range from 0 to 1.
                point = (
                    int(landmark.x * width),
                    int(landmark.y * height),
                )

                # Save the point so it can be connected later.
                points.append(point)

                # Draw a purple dot on the hand landmark.
                cv2.circle(
                    frame,
                    point,
                    4,
                    (255, 0, 255),  # Purple in BGR format
                    -1,             # Filled circle
                )

            # Draw lines between connected landmark points.
            for start_index, end_index in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (255, 0, 255),
                    2,
                )

        # -------------------- SHOW COMBINED FPS --------------------

        # Calculate frames processed per second.
        # This includes both EfficientDet and hand tracking.
        fps = 1 / (time.perf_counter() - start)

        # Display FPS in yellow in the upper-left corner.
        cv2.putText(
            frame,
            f"Combined FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Show the final annotated webcam frame.
        cv2.imshow("Hands + EfficientDet Hardware Test", frame)

        # Press q to exit the program.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Release the webcam so other programs can use it.
camera.release()

# Close the OpenCV window.
cv2.destroyAllWindows()