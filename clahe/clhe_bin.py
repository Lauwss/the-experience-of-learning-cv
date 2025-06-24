import numpy as np
import cv2
import matplotlib.pyplot as plt

def redistribute_histogram_absolute(img, clip_limit=2000, show_stats=True):
    h, w = img.shape
    n_pixels = h * w

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()  # shape: (256,)
    total_excess = np.sum(np.maximum(hist - clip_limit, 0))
    clipped_hist = np.minimum(hist, clip_limit)
    redistributed_hist = clipped_hist + total_excess / 256

    # 概率密度 & CDF
    redistributed_prob = redistributed_hist / n_pixels
    cdf = np.cumsum(redistributed_prob)
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)

    # 查找表 LUT
    mapping = cdf_scaled

    # 应用映射
    result_img = mapping[img]

    if show_stats:
        print(f"\n{'灰度':>5} | {'频数':>8} | {'新频数':>10} | {'新概率':>8} | {'累积概率':>10}")
        print("-" * 60)
        for i in range(256):
            if hist[i] > 0:
                print(f"{i:5d} | {int(hist[i]):8d} | {int(redistributed_hist[i]):10d} | {redistributed_prob[i]:8.4f} | {cdf[i]:10.4f}")

    return result_img, redistributed_prob, cdf

# ===================== 主流程 =====================

img = cv2.imread(r'C:\Users\20342\Desktop\cv\clahe\original_dark_image.png', cv2.IMREAD_GRAYSCALE)

# 使用绝对频数阈值进行直方图裁剪
adjusted_img, new_prob, new_cdf = redistribute_histogram_absolute(img, clip_limit=2000)

# 可视化
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(adjusted_img, cmap='gray')
plt.title("After Histogram Equalization (Clipped)")
plt.axis('off')

# 原图直方图
plt.subplot(2, 2, 3)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, label='Original Histogram', color='gray')
plt.title("Original Histogram")
plt.grid(True)

# 新的直方图
plt.subplot(2, 2, 4)
plt.plot(new_prob * img.size, label='Clipped + Redistributed', color='green')
plt.plot(new_cdf * img.size, label='CDF', color='red')
plt.title("Modified Histogram + CDF")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
