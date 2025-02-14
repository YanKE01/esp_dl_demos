import cv2
import numpy as np

# Read the image
image_path = "./asset/img.png"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Please check the path.")
    exit()

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow("HSV Adjustments", cv2.WINDOW_AUTOSIZE)

# Initialize HSV thresholds
lower_red1 = [0, 120, 70]
upper_red1 = [10, 255, 255]
lower_red2 = [170, 120, 70]
upper_red2 = [180, 255, 255]


# Callback function for trackbars
def nothing(x):
    pass


# Create HSV range adjustment trackbars
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

while True:
    # Get trackbar positions
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

    # Generate masks
    mask1 = cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1))
    mask2 = cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
    mask = mask1 | mask2  # Merge the two masks

    # Show the mask
    cv2.imshow("Mask", mask)

    # Print current HSV values (useful for copying into C++ code)
    print(f"lower_red1 = cv::Scalar({lower_red1[0]}, {lower_red1[1]}, {lower_red1[2]});")
    print(f"upper_red1 = cv::Scalar({upper_red1[0]}, {upper_red1[1]}, {upper_red1[2]});")
    print(f"lower_red2 = cv::Scalar({lower_red2[0]}, {lower_red2[1]}, {lower_red2[2]});")
    print(f"upper_red2 = cv::Scalar({upper_red2[0]}, {upper_red2[1]}, {upper_red2[2]});\n")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
