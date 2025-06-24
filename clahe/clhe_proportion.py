import numpy as np
import cv2
import matplotlib.pyplot as plt

def redistribute_histogram(img, threshold=0.005, show_stats=True):
    h, w = img.shape
    n_pixels = h * w

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    prob = hist / n_pixels

    # Step 1: 找出超出阈值的灰度（flatten）
    excess = np.maximum(prob - threshold, 0)
    kept = np.minimum(prob, threshold)
    total_excess = excess.sum()

    # Step 2: 均匀重分布
    redistributed = kept + total_excess / 256

    # Step 3: 累积分布函数
    cdf = np.cumsum(redistributed)
    cdf_scaled = np.round(cdf * 255).astype(np.uint8)

    # Step 4: 构建查找表（LUT）
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = cdf_scaled[i]

    # Step 5: 应用映射
    result_img = mapping[img]


    if show_stats:
            print(f"\n{'灰度':>5} | {'新概率':>8} | {'累积概率':>10} | {'频数(像素)':>10}")
            print("-" * 45)
            for i in range(256):
                count = int(redistributed[i] * n_pixels)
                if count > 0:
                    print(f"{i:5d} | {redistributed[i]:8.4f} | {cdf[i]:10.4f} | {count:10d}")      
    return result_img, redistributed, cdf


# ===================== 主流程 =====================

img = cv2.imread(r'C:\Users\20342\Desktop\cv\clahe\original_dark_image.png', cv2.IMREAD_GRAYSCALE)

# 直方图平滑 + 自定义均衡化
adjusted_img, new_prob, new_cdf = redistribute_histogram(img)

# 可视化原图与处理图
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(adjusted_img, cmap='gray')
plt.title("Adjusted HE (Flattened Histogram)")
plt.axis('off')

# 原图直方图
plt.subplot(2, 2, 3)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, label='Original Histogram', color='gray')
plt.title("Original Histogram")
plt.grid(True)

# 调整后直方图
plt.subplot(2, 2, 4)
plt.plot(new_prob * img.size, label='Flattened Histogram', color='green')
plt.plot(new_cdf * img.size, label='CDF', color='red')
plt.title("Modified Histogram + CDF")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
