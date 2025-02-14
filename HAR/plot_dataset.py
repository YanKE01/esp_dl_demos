import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./dataset/train.csv')

activity_counts = train['Activity'].value_counts()
print("\nDistribution of activity labels in the training dataset:")
print(activity_counts)

plt.figure(figsize=(12, 8))  # 可以调大图表尺寸
activity_counts.plot(kind='bar')

plt.title('Activity Distribution')
plt.xlabel('Activity')
plt.ylabel('Count')

# 调整 X 轴标签的字体大小和旋转角度
plt.xticks(rotation=45, fontsize=10)  # 旋转标签并减小字体
plt.tight_layout()  # 防止标签被截断

plt.show()
