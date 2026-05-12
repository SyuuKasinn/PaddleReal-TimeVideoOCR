import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from collections import defaultdict
import re
import MeCab
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import jaconv
from rapidocr_onnxruntime import RapidOCR

# ストップワードのダウンロード
nltk.download('stopwords')

# MeCabとPaddleOCRの初期化
m = MeCab.Tagger("-Owakati")

model = RapidOCR(rec_model_path=r"C:\syuu\pythonProject1\japan_PP-OCRv3_rec_infer.onnx")
ocr = RapidOCR(text_score=0.85, use_gpu=True, det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)

# レンズ歪み補正パラメータ（これらのパラメータはサンプル値であり、実際の較正結果に基づいて調整する必要があります）
K = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]])  # 内部パラメータ行列
D = np.array([0, 0, 0, 0])  # 歪み係数


def adjust_clahe_params(image, clip_limit_range=(1.0, 10.0), tile_grid_size_range=(2, 10)):
    if image is None or not (len(image.shape) == 3 and image.shape[2] == 3):
        raise ValueError("入力はBGR画像である必要があります。")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    brightness = np.mean(gray)

    clip_limit = min(max(contrast / 10.0, clip_limit_range[0]), clip_limit_range[1])

    height, width = gray.shape
    avg_size = (height + width) / 2.0
    tile_grid_size = min(max(int(avg_size / 2000.0), tile_grid_size_range[0]), tile_grid_size_range[1])

    if brightness < 80:
        tile_grid_size = min(tile_grid_size + 1, tile_grid_size_range[1])

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    equalized_image = clahe.apply(gray)

    return equalized_image


def detail_enhance(image, sigma_s=15, sigma_r=0.2):
    return cv2.detailEnhance(image, sigma_s, sigma_r)


def adjust_hsv_v_channel(image, v_increase=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + v_increase, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def automatic_gaussian_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = adjust_clahe_params(image)
    kernel_size = max(1, math.floor(gray.shape[1] / 20) | 1)
    return cv2.GaussianBlur(clahe_image, (kernel_size, kernel_size), 0)


def draw_text(image, text, position):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font_path = '../M_PLUS_1p/MPLUS1p-Regular.ttf'
    font = ImageFont.truetype(font_path, 20)
    draw.text(position, text, font=font, fill=(0, 0, 255, 255))
    return np.array(img_pil)


def calculate_iou(box1, box2):
    box1, box2 = np.array(box1), np.array(box2)
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    min_x, min_y = min(box1[0], box2[0]), min(box1[1], box2[1])
    max_x, max_y = max(box1[2], box2[2]), max(box1[3], box2[3])
    c_area = (max_x - min_x) * (max_y - min_y)
    iou = inter_area / union_area
    return iou - (c_area - union_area) / c_area


def preprocess_japanese_text(text):
    # 日本語テキストの正規化とフィルタリング
    def normalize_japanese_text(text):
        # 全角文字を半角に変換
        text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
        # ひらがなをカタカナに変換
        text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
        return text

    filtered_text = normalize_japanese_text(text)
    return filtered_text


def analyze_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    min_w = 10
    min_h = 10
    max_area = 10000

    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h >= min_area and (h >= min_h or w >= min_w) and w * h <= max_area):
            filtered_contours.append(cnt)
        else:
            gray[y:y + h, x:x + w] = 0
    return cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 0), 3)


def get_contour(areas, contours, isDark):
    contour = contours[np.argmax(areas)]
    return contour, isDark


def get_maximum_uniform_contour(image, fontsize, margin=0):
    """画像で最大の一様な輪郭を検出"""
    if margin > 0:
        image = image[margin:-margin, margin:-margin]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(src=gray, ksize=(fontsize // 2, fontsize // 2))
    _, threshold = cv2.threshold(src=blurred, thresh=255 / 2, maxval=255, type=cv2.THRESH_BINARY)
    contoursDark = cv2.findContours(255 - threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
    contoursLight = cv2.findContours(threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[-2]

    areasDark = [cv2.contourArea(cnt) for cnt in contoursDark]
    areasLight = [cv2.contourArea(cnt) for cnt in contoursLight]

    if not areasDark and not areasLight:
        return None, None

    maxDarkArea = max(areasDark) if areasDark else 0
    maxLightArea = max(areasLight) if areasLight else 0

    if maxDarkArea < (4 * fontsize) ** 2 and maxLightArea < (4 * fontsize) ** 2:
        return None, None

    contour, isDark = None, None

    if maxLightArea < maxDarkArea:
        contour, isDark = get_contour(areasDark, contoursDark, True)
    else:
        contour, isDark = get_contour(areasLight, contoursLight, False)

    if contour is not None:
        contour += margin

    return contour, isDark


def correct_lens_distortion(image, K, D):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, D, None, new_camera_matrix)
    return undistorted_img


def main():
    start_time = time.time()
    try:
        image_path = r'C:\syuu\pythonProject\IMG_1867.JPG'
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"画像ファイル '{image_path}' が見つからないか、読み込むことができません。")

        # レンズ歪み補正
        image = correct_lens_distortion(image, K, D)

        # 画像のリサイズ
        height, width = image.shape[:2]
        new_size_half = (width // 2, height // 2)
        image_resized_half = cv2.resize(image, new_size_half, interpolation=cv2.INTER_LINEAR)

        # 最大輪郭の検出
        fontsize = 20
        contour, isDark = get_maximum_uniform_contour(image_resized_half, fontsize, margin=10)

        if contour is not None:
            cv2.drawContours(image_resized_half, [contour], -1, (0, 255, 0), 3)

        roi_image = analyze_roi(image_resized_half)

        # 画像の強調
        detail_enhanced = detail_enhance(roi_image)
        blurred = automatic_gaussian_blur(detail_enhanced)
        denoised_gray = cv2.fastNlMeansDenoising(blurred, h=10, templateWindowSize=7, searchWindowSize=21)

        # エッジ検出
        canny_edges = cv2.Canny(denoised_gray, 30, 150)
        sobel_x = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(denoised_gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
        laplacian_edges = cv2.convertScaleAbs(cv2.Laplacian(denoised_gray, cv2.CV_64F))

        edged = np.maximum.reduce([canny_edges, sobel_edges, laplacian_edges])
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

        # OCRとテキスト検出
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        results_list = []

        for scale in scales:
            scaled_img = cv2.resize(image_with_contours.copy(), None, fx=scale, fy=scale)
            result = ocr(scaled_img)
            temp_list = list(result)
            temp_list.pop()
            result = tuple(temp_list)
            for line in result:
                for word_info in line:
                    word = preprocess_japanese_text(word_info[1])
                    confidence = word_info[2]
                    if confidence >= 0.8:
                        coordinates = np.array(word_info[0])
                        min_coordinates = coordinates.min(axis=0) / scale
                        max_coordinates = coordinates.max(axis=0) / scale
                        x_min, y_min = min_coordinates
                        x_max, y_max = max_coordinates
                        results_list.append((word, [x_min, y_min, x_max, y_max], confidence))

        # 非極大抑制
        results_list.sort(key=lambda x: -x[2])
        keep = [1] * len(results_list)

        for i in range(len(results_list)):
            if keep[i] == 0:
                continue
            _, box1, _ = results_list[i]
            for j in range(i + 1, len(results_list)):
                _, box2, _ = results_list[j]
                if calculate_iou(box1, box2) > 0.25:
                    keep[j] = 0

        # 結果を画像に描画
        result_img = image.copy()
        for k, (word, box, _) in enumerate(results_list):
            if keep[k] == 1:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                result_img = draw_text(result_img, word, (x_min, y_min - 5))

        # 画像を表示
        cv2.imshow("Original Image", image)
        cv2.imshow("Blurred Image", blurred)
        cv2.imshow("Edge Detection Image", edged)
        cv2.imshow("Laplacian Edge Detection", laplacian_edges)
        cv2.imshow("OCR Result Image", result_img)
        cv2.imshow("ROI Image", roi_image)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    end_time = time.time()
    print(f'総実行時間: {end_time - start_time:.2f} 秒')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
