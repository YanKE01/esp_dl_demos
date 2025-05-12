import cv2
import re

# Input image path and YOLO inference log string
image_path = "asset/bus.jpg"
log_str = """
I (2288) yolo11n: [category: 5, score: 0.679179, x1: 6, y1: 110, x2: 400, y2: 369]
I (2288) yolo11n: [category: 0, score: 0.562177, x1: 21, y1: 195, x2: 108, y2: 452]
I (2298) yolo11n: [category: 0, score: 0.377541, x1: 112, y1: 204, x2: 164, y2: 429]
I (2308) yolo11n: [category: 0, score: 0.377541, x1: 334, y1: 183, x2: 403, y2: 435]
"""

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    exit()

# Regular expression to extract detection results
pattern = r"\[category: (\d+), score: ([\d.]+), x1: (\d+), y1: (\d+), x2: (\d+), y2: (\d+)\]"
matches = re.findall(pattern, log_str)

# Draw bounding boxes and labels
for match in matches:
    category, score, x1, y1, x2, y2 = match
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    category = int(category)
    score = float(score)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label text
    label = f"Class {category}: {score:.2f}"
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the image
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
