import cv2
import numpy as np

# Set to False to use an image, True to use video stream (e.g., webcam or video file)
use_video = True

# If using an image
image_path = "./asset/img.png"  # Replace with your image path

# If using video, you can use 0 for webcam or provide a file path
video_source = 0  # 0 = default webcam; or use "video.mp4"

# Create a window for HSV trackbars
cv2.namedWindow("HSV Adjustments", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Adjustments", 400, 600)

# Initial HSV threshold values for red color detection
lower_red1 = [0, 120, 70]
upper_red1 = [10, 255, 255]
lower_red2 = [170, 120, 70]
upper_red2 = [180, 255, 255]

# Callback for trackbar (required but unused)
def nothing(x):
    pass

# Create trackbars for HSV range adjustments
cv2.createTrackbar("Low H1", "HSV Adjustments", lower_red1[0], 180, nothing)
cv2.createTrackbar("Low S1", "HSV Adjustments", lower_red1[1], 255, nothing)
cv2.createTrackbar("Low V1", "HSV Adjustments", lower_red1[2], 255, nothing)
cv2.createTrackbar("High H1", "HSV Adjustments", upper_red1[0], 180, nothing)
cv2.createTrackbar("High S1", "HSV Adjustments", upper_red1[1], 255, nothing)
cv2.createTrackbar("High V1", "HSV Adjustments", upper_red1[2], 255, nothing)

cv2.createTrackbar("Low H2", "HSV Adjustments", lower_red2[0], 180, nothing)
cv2.createTrackbar("Low S2", "HSV Adjustments", lower_red2[1], 255, nothing)
cv2.createTrackbar("Low V2", "HSV Adjustments", lower_red2[2], 255, nothing)
cv2.createTrackbar("High H2", "HSV Adjustments", upper_red2[0], 180, nothing)
cv2.createTrackbar("High S2", "HSV Adjustments", upper_red2[1], 255, nothing)
cv2.createTrackbar("High V2", "HSV Adjustments", upper_red2[2], 255, nothing)

# Use image mode
if not use_video:
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Please check the path.")
        exit()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Use video stream mode
else:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Failed to open video source.")
        exit()

# Main loop
while True:
    # For video: read the current frame
    if use_video:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video.")
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        frame = image  # Static image stays the same
        # HSV was already calculated for image mode

    # Read HSV values from trackbars
    lower_red1 = [cv2.getTrackbarPos("Low H1", "HSV Adjustments"),
                  cv2.getTrackbarPos("Low S1", "HSV Adjustments"),
                  cv2.getTrackbarPos("Low V1", "HSV Adjustments")]
    upper_red1 = [cv2.getTrackbarPos("High H1", "HSV Adjustments"),
                  cv2.getTrackbarPos("High S1", "HSV Adjustments"),
                  cv2.getTrackbarPos("High V1", "HSV Adjustments")]
    lower_red2 = [cv2.getTrackbarPos("Low H2", "HSV Adjustments"),
                  cv2.getTrackbarPos("Low S2", "HSV Adjustments"),
                  cv2.getTrackbarPos("Low V2", "HSV Adjustments")]
    upper_red2 = [cv2.getTrackbarPos("High H2", "HSV Adjustments"),
                  cv2.getTrackbarPos("High S2", "HSV Adjustments"),
                  cv2.getTrackbarPos("High V2", "HSV Adjustments")]

    # Generate masks for the red color range
    mask1 = cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1))
    mask2 = cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
    mask = mask1 | mask2  # Combine both masks

    # Display the mask and original image/video
    cv2.imshow("Mask", mask)
    cv2.imshow("Video/Image", frame)

    # Print the current HSV values (for reference or use in C++/OpenCV)
    print(f"lower_red1 = cv::Scalar({lower_red1[0]}, {lower_red1[1]}, {lower_red1[2]});")
    print(f"upper_red1 = cv::Scalar({upper_red1[0]}, {upper_red1[1]}, {upper_red1[2]});")
    print(f"lower_red2 = cv::Scalar({lower_red2[0]}, {lower_red2[1]}, {lower_red2[2]});")
    print(f"upper_red2 = cv::Scalar({upper_red2[0]}, {upper_red2[1]}, {upper_red2[2]});\n")

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture if using video
if use_video:
    cap.release()

cv2.destroyAllWindows()
