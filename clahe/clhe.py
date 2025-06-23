import numpy as np
import cv2
import matplotlib.pyplot as plt


# 图像直方图可视化
def plot_histogram(img, title="Histogram"):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='gray')
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.xlim([0, 256])
    plt.grid(True)

# 生成暗图
dark_img =cv2.imread(r'C:\Users\20342\Desktop\cv\clahe\original_dark_image.png', cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡化
he_img = cv2.equalizeHist(dark_img)

# 可视化结果
plt.figure(figsize=(12, 8))

# 原图
plt.subplot(2, 2, 1)
plt.imshow(dark_img, cmap='gray')
plt.title("Original Dark Image")
plt.axis('off')

# 均衡化后图像
plt.subplot(2, 2, 2)
plt.imshow(he_img, cmap='gray')
plt.title("After Histogram Equalization")
plt.axis('off')

# 原图直方图
plt.subplot(2, 2, 3)
plot_histogram(dark_img, "Original Histogram")

# HE 后直方图
plt.subplot(2, 2, 4)
plot_histogram(he_img, "Equalized Histogram")

plt.tight_layout()
plt.show()
