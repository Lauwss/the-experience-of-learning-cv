import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像直方图 + 输出统计信息
def plot_histogram(img, title="Histogram", show_stats=False):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    prob_density = hist / hist.sum()
    cdf = np.cumsum(prob_density)

    plt.plot(hist, color='gray', label="Histogram")
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.xlim([0, 256])
    plt.grid(True)

    # 叠加累积分布曲线（可视化）
    plt.twinx()
    plt.plot(cdf, color='red', label="CDF")
    plt.ylabel('Cumulative Probability')
    plt.ylim([0, 1])
    
    # 输出统计数据（可选）
    if show_stats:
        print(f"\n===== {title} =====")
        print("灰度值\t像素数\t概率\t累积概率")
        for i in range(256):
            if hist[i] > 0:
                print(f"{i:3}\t{int(hist[i])}\t{prob_density[i]:.4f}\t{cdf[i]:.4f}")

# 读取图像（灰度）
dark_img = cv2.imread(r'C:\Users\20342\Desktop\cv\clahe\original_dark_image.png', cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡化
he_img = cv2.equalizeHist(dark_img)

# 可视化
plt.figure(figsize=(12, 8))

# 原图
plt.subplot(2, 2, 1)
plt.imshow(dark_img, cmap='gray')
plt.title("Original Dark Image")
plt.axis('off')

# HE 后图像
plt.subplot(2, 2, 2)
plt.imshow(he_img, cmap='gray')
plt.title("After Histogram Equalization")
plt.axis('off')

# 原图直方图 + 统计输出
plt.subplot(2, 2, 3)
plot_histogram(dark_img, "Original Histogram", show_stats=True)

# HE 后直方图 + 统计输出
plt.subplot(2, 2, 4)
plot_histogram(he_img, "Equalized Histogram", show_stats=True)

plt.tight_layout()
plt.show()
