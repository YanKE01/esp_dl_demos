import cv2
import numpy as np

image_path = "asset/img.png"
image = cv2.imread(image_path)

if image is None:
    exit()

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV color ranges for red
lower_red1 = np.array([0, 40, 149])
upper_red1 = np.array([17, 193, 255])
lower_red2 = np.array([174, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Generate binary masks
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and draw enclosing circles and rectangles
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Filter out small noise
        # Calculate bounding rectangle
        x, y, w, h = 200, 0, 408, 288
        print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue rectangle

# Display result
cv2.imshow("Red Ball Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
