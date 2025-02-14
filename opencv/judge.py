import cv2
import numpy as np

# 读取图片
image_path = "./asset/img_1.png"  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径")
    exit()

# 转换到 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建窗口
cv2.namedWindow("HSV Adjustments", cv2.WINDOW_AUTOSIZE)

# 初始化 HSV 阈值
lower_red1 = [0, 120, 70]
upper_red1 = [10, 255, 255]
lower_red2 = [170, 120, 70]
upper_red2 = [180, 255, 255]

# 轨迹条回调函数
def nothing(x):
    pass

# 创建 HSV 颜色范围调整轨迹条
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
    # 获取轨迹条的值
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

    # 生成掩码
    mask1 = cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1))
    mask2 = cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
    mask = mask1 | mask2  # 合并两个掩码

    # 显示掩码
    cv2.imshow("Mask", mask)

    # 打印当前 HSV 值（便于复制到 C++ 代码）
    print(f"lower_red1 = cv::Scalar({lower_red1[0]}, {lower_red1[1]}, {lower_red1[2]});")
    print(f"upper_red1 = cv::Scalar({upper_red1[0]}, {upper_red1[1]}, {upper_red1[2]});")
    print(f"lower_red2 = cv::Scalar({lower_red2[0]}, {lower_red2[1]}, {lower_red2[2]});")
    print(f"upper_red2 = cv::Scalar({upper_red2[0]}, {upper_red2[1]}, {upper_red2[2]});\n")

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
