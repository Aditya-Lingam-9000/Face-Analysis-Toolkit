import os                      # For file/folder operations
import argparse                # For command-line arguments

import cv2                     # OpenCV for image & video processing
import mediapipe as mp         # MediaPipe for face detection


def process_img(img, face_detection):
    """
    Detects faces in an image/frame and blurs them.
    """

    H, W, _ = img.shape         # Image height & width (needed to convert coords)

    # MediaPipe expects RGB, OpenCV gives BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face detection
    results = face_detection.process(img_rgb)

    # Proceed only if faces are detected
    if results.detections:
        for detection in results.detections:

            # Get relative bounding box (values between 0 and 1)
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coords to pixel coords
            x = int(bbox.xmin * W)
            y = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Blur only the face region (face anonymization)
            img[y:y+h, x:x+w] = cv2.blur(
                img[y:y+h, x:x+w],
                (30, 30)
            )

    return img


# -------------------- Argument Parsing --------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="webcam")      # image | video | webcam
parser.add_argument("--filePath", default=None)      # path for image/video
args = parser.parse_args()


# -------------------- Output Directory --------------------

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)   # Safe folder creation


# -------------------- MediaPipe Face Detector --------------------

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
        model_selection=0,               # Short-range face model
        min_detection_confidence=0.5     # Ignore weak detections
) as face_detection:


    # -------------------- IMAGE MODE --------------------
    if args.mode == "image":

        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)

        cv2.imwrite(
            os.path.join(output_dir, "output.png"),
            img
        )


    # -------------------- VIDEO MODE --------------------
    elif args.mode == "video":

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        video_writer = cv2.VideoWriter(
            os.path.join(output_dir, "output.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"),
            25,
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            frame = process_img(frame, face_detection)
            video_writer.write(frame)
            ret, frame = cap.read()

        cap.release()
        video_writer.release()


    # -------------------- WEBCAM MODE --------------------
    elif args.mode == "webcam":

        cap = cv2.VideoCapture(2)     # Change index if webcam not detected
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow("Webcam Face Blur", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
