import re
import cv2

# Path to the image (replace with your actual image path)
image_path = 'asset/test.jpg'

# Read the image
image = cv2.imread(image_path)

# Detection result string
detection_str = "I (2845) yolo11n: [category: 0, score: 0.777300, x1: 100, y1: 82, x2: 371, y2: 363]"

# Parse the detection string using regular expressions
pattern = r"category:\s*(\d+),\s*score:\s*([\d\.]+),\s*x1:\s*(\d+),\s*y1:\s*(\d+),\s*x2:\s*(\d+),\s*y2:\s*(\d+)"
match = re.search(pattern, detection_str)
if not match:
    raise ValueError("Failed to parse detection string")

# Extract values and convert to appropriate types
category = int(match.group(1))
score = float(match.group(2))
x1, y1, x2, y2 = map(int, match.groups()[2:])

# Draw a rectangle on the image
color = (0, 255, 0)    # Green color in BGR
thickness = 2
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Create and place the label above the rectangle
label = f'Category: {category}, Score: {score:.2f}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
line_type = cv2.LINE_AA
cv2.putText(image, label, (x1, y1 - 10), font, font_scale, color, 1, line_type)

# Display the result in a window
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save the image with the drawn box
# cv2.imwrite('output.jpg', image)
