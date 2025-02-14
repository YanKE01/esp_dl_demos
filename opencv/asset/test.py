import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 背景尺寸
bg_width = 1024
bg_height = 600

# 目标框的坐标和尺寸
x, y, width, height = 677, 305, 43, 37

# 创建背景图
fig, ax = plt.subplots(figsize=(bg_width / 100, bg_height / 100))  # 单位为英寸，1英寸=100像素

# 设置背景颜色
ax.set_facecolor('lightgrey')

# 画一个矩形框来表示检测到的红色物体
rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# 设置坐标轴范围
ax.set_xlim(0, bg_width)
ax.set_ylim(0, bg_height)

# 添加注释
ax.text(x + width + 10, y + height / 2, 'Red Object Detected', fontsize=12, color='red')

# 隐藏坐标轴
ax.axis('off')

# 显示图片
plt.show()
