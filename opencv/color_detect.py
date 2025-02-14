import cv2
import numpy as np

# 读取图像
image_path = "./asset/img_2.png"  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径")
    exit()

# 转换为 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的 HSV 颜色范围
lower_red1 = np.array([0, 40, 149])
upper_red1 = np.array([17, 193, 255])
lower_red2 = np.array([174, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 生成二值化掩码
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 形态学操作去除噪声
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓并绘制外接圆和矩形
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # 过滤掉小噪点
        # 计算矩形边界框
        x, y, w, h = 200,0,408,288
        print(x,y,w,h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 画蓝色矩形框

# 显示结果
cv2.imshow("Red Ball Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
