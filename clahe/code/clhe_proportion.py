import numpy as np
import cv2
import matplotlib.pyplot as plt

def redistribute_histogram(img, threshold=0.005):
    h, w = img.shape
    n_pixels = h * w

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    prob = hist / n_pixels

    # 截断与平滑
    excess = np.maximum(prob - threshold, 0)
    kept = np.minimum(prob, threshold)
    total_excess = excess.sum()

    redistributed = kept + total_excess / 256
    cdf = np.cumsum(redistributed)
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)

    # 映射并返回结果图像
    mapping = cdf_scaled
    result_img = mapping[img]

    return result_img

# ===================== 主流程 =====================

# 读取图像
img = cv2.imread(r'C:\Users\20342\Desktop\cv\clahe\code\original_dark_image.png', cv2.IMREAD_GRAYSCALE)

# 阈值列表
thresholds = [0.1, 0.01, 0.001]
titles = [f"Threshold = {t}" for t in thresholds]

# 可视化图像结果
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

# 不同阈值的均衡化结果
for i, t in enumerate(thresholds):
    adjusted_img = redistribute_histogram(img, threshold=t)
    plt.subplot(1, 4, i + 2)
    plt.imshow(adjusted_img, cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
