import math
import multiprocessing
import re
import threading
import tkinter as tk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import MeCab
import matplotlib.pyplot as plt
import pygame
import torch
from fuzzywuzzy import fuzz
from jaconv import jaconv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import spacy
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from paddleocr import PaddleOCR
import os

from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, BertJapaneseTokenizer

from acceptedWords import ocr_to_accepted_words

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 環境変数の設定
nlp = spacy.load('ja_core_news_md')  # SpaCyの日本語モデルを読み込み
ocr_to_accepted_words = ocr_to_accepted_words


def draw_transparent_rectangle(image, top_left, bottom_right, color, alpha=0.5):
    """
    画像上に半透明の長方形を描きます。

    パラメータ:
    - 画像: オリジナル画像 (BGR形式)。
    - 左上: 長方形の左上の頂点 (x, y)。
    - 右下: 長方形の右下の頂点 (x, y)。
    - 色: 長方形の色 (BGR形式)。
    - 透明度: 透明度レベル (0.0 から 1.0)。
    """
    # オリジナル画像と同じサイズのオーバーレイ画像を作成します
    overlay = image.copy()

    # 長方形をオーバーレイに描きます
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)

    # オーバーレイをオリジナル画像と合成します
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


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


def correct_lens_distortion(image, K, D):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, D, None, new_camera_matrix)
    return undistorted_img


def apply_canny(eroded):
    return cv2.Canny(eroded, 30, 150)


def apply_sobel(blurred):
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel_edges)


def apply_laplacian(blurred):
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)


def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def compute_sample_bounds(image, sample_fraction):
    height, width = image.shape
    start_row = int((1 - sample_fraction) / 2 * height)
    end_row = start_row + int(sample_fraction * height)
    start_col = int((1 - sample_fraction) / 2 * width)
    end_col = start_col + int(sample_fraction * width)
    return start_row, start_col, end_row, end_col


def get_sample_stats(image, sample_fraction=0.5):
    try:
        # Calculate the size of the sample region
        start_row, start_col, end_row, end_col = compute_sample_bounds(image, sample_fraction)

        # Extract and compute statistics for the sample region
        sample = image[start_row:end_row, start_col:end_col]
        mean = np.mean(sample)
        stddev = np.std(sample)

        return mean, stddev
    except Exception as e:
        raise Exception("Error in get_sample_stats: {}".format(e))


def compute_clahe_params(stddev, clip_limit_range, height, width, tile_grid_size_range):
    clip_limit = min(max(stddev / 10.0, clip_limit_range[0]), clip_limit_range[1])
    tile_grid_size = min(max(int((height + width) / 2000), tile_grid_size_range[0]),
                         tile_grid_size_range[1])
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))


def should_recalculate_clahe(mean, stddev, prev_mean, prev_stddev, clahe_threshold):
    return abs(mean - prev_mean) >= clahe_threshold or abs(stddev - prev_stddev) >= clahe_threshold


def adjust_clahe_params(self, image, clip_limit_range=(1.0, 10.0), tile_grid_size_range=(2, 10)):
    try:
        mean, stddev = get_sample_stats(image, sample_fraction=0.5)
        height, width = image.shape
        clahe = None

        if self.previous_img_stats is None:
            clahe = compute_clahe_params(stddev, clip_limit_range, height, width, tile_grid_size_range)
        else:
            prev_mean, prev_stddev = self.previous_img_stats
            if should_recalculate_clahe(mean, stddev, prev_mean, prev_stddev, self.clahe_threshold):
                clahe = compute_clahe_params(stddev, clip_limit_range, height, width, tile_grid_size_range)
            else:
                clahe = self.previous_clahe

        self.previous_img_stats = (mean, stddev)
        self.previous_clahe = clahe

        return clahe.apply(image)
    except Exception as e:
        print(f"Error occurred while adjusting CLAHE parameters: {e}")
        return image


def automatic_gaussian_blur(image):
    try:
        if len(image.shape) == 2:  # 画像がグレースケールの場合
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # グレースケールからBGRに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 画像をグレースケールに変換
        kernel_size = math.floor(gray.shape[1] / 20)  # カーネルサイズの計算
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)  # カーネルサイズを奇数に調整
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  # ガウスぼかしの適用
        return blurred
    except Exception as e:
        print(f"ガウスぼかしの適用中にエラーが発生しました: {e}")
        return image


def load_font(font_path, font_size):
    return ImageFont.truetype(font_path, font_size)


def calculate_text_size(font, text):
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    return text_width, text_height


def draw_background(draw, position, text_width, text_height, bg_color, padding):
    x, y = position
    rect_x1 = x - padding
    rect_y1 = y - text_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + padding
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=bg_color)


def draw_text_on_draw(draw, position, text, font, text_color, text_height, padding):
    x, y = position
    draw.text((x, y - text_height - padding), text, font=font, fill=text_color)


def draw_text(image, text, position, font_path='M_PLUS_1p/MPLUS1p-Regular.ttf', font_size=20,
              text_color=(0, 0, 255), bg_color=(255, 255, 255), padding=5):
    try:
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        font = load_font(font_path, font_size)
        text_width, text_height = calculate_text_size(font, text)
        draw_background(draw, position, text_width, text_height, bg_color, padding)
        draw_text_on_draw(draw, position, text, font, text_color, text_height, padding)

        return np.array(img_pil)
    except Exception as e:
        print(f"Error occurred: {e}")
        return image


def calculate_image_iou(box1, box2):
    xi1, yi1, xi2, yi2 = np.maximum(box1[0], box2[0]), np.maximum(box1[1], box2[1]), np.minimum(box1[2],
                                                                                                box2[2]), np.minimum(
        box1[3], box2[3])
    inter_area = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)  # 交差部分の面積を計算
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  # ボックス1の面積を計算
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  # ボックス2の面積を計算
    union_area = box1_area + box2_area - inter_area  # 結合部分の面積を計算
    return inter_area / union_area  # IOUを返す


def analyze_roi(gray):
    # Convert the image to grayscale

    # Use thresholding to identify the regions of interest
    # Here I use Otsu's binarization method for thresholding.
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)

    # Draw contours around the defined ROIs
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_image = cv2.drawContours(gray.copy(), contours, -1, (0, 255, 0), 3)

    return roi_image




def calculate_video_ciou(box1, box2):
    # ボックスの座標を展開します
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # 交差する座標を計算します
    xi1 = np.maximum(x1, x1_)
    yi1 = np.maximum(y1, y1_)
    xi2 = np.minimum(x2, x2_)
    yi2 = np.minimum(y2, y2_)

    # 交差する領域の面積を計算します
    inter_area = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)

    # 各ボックスの面積を計算します
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # 和集合の面積を計算します
    union_area = box1_area + box2_area - inter_area

    # IoU (Intersection over Union) を計算します
    iou = inter_area / union_area if union_area != 0 else 0

    # 最小の外接矩形の座標を計算します
    min_x = np.minimum(x1, x1_)
    min_y = np.minimum(y1, y1_)
    max_x = np.maximum(x2, x2_)
    max_y = np.maximum(y2, y2_)

    # 最小の外接矩形の面積を計算します
    c_area = (max_x - min_x) * (max_y - min_y)

    # GIoU (Generalized Intersection over Union) を計算します
    giou = iou - (c_area - union_area) / c_area if c_area != 0 else iou - 1

    # 中心の距離を計算します
    center1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    center2 = np.array([(x1_ + x2_) / 2, (y1_ + y2_) / 2])
    center_dist = np.linalg.norm(center1 - center2)

    # 最小の外接矩形の対角線の長さを計算します
    diag_length = np.linalg.norm([max_x - min_x, max_y - min_y])

    # アスペクト比を計算します
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x2_ - x1_, y2_ - y1_
    aspect_ratio1 = w1 / h1 if h1 != 0 else 0
    aspect_ratio2 = w2 / h2 if h2 != 0 else 0
    aspect_ratio_dist = (aspect_ratio1 - aspect_ratio2) ** 2

    # CIoU (Complete Intersection over Union) を計算します
    alpha = aspect_ratio_dist / (1 - iou + 1e-10)  # ゼロ除算を避けるために小さな定数を追加します
    ciou = giou - (center_dist / (diag_length + 1e-10)) - alpha  # ゼロ除算を避けるために小さな定数を追加します

    return ciou


# TODO: Use Path for all file paths
from pathlib import Path

class App:
    def __init__(self, window, window_title, video_source=0):

        cpu_cores = multiprocessing.cpu_count()  # 获取CPU核数

        self.window = window  # Tkinterウィンドウの設定
        # self.window.title(window_title)  # ウィンドウのタイトル設定
        self.video_source = video_source  # ビデオソースの設定
        self.vid = None  # ビデオキャプチャオブジェクトの初期化
        self.ocr_en = PaddleOCR(use_angle_cls=True, lang='japan', enable_mkldnn=True, use_gpu=True)  # PaddleOCRの日本語モデルの設定
        self.result_label = tk.Label(window, text="光学文字認識")  # OCR結果を表示するラベルの作成
        self.result_label.pack()  # ラベルをウィンドウに配置
        self.camera_open = False  # カメラのオープン状態の初期化
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_cores)  # スレッドプールの設定

        # ボタンの設定と配置
        self.buttons = {
            "カメラ": self.open_camera,
            "スナップショット": self.take_snapshot,
            "OCR": self.perform_ocr,
            "終了": self.close
        }

        self.button_widgets = {}
        for name, action in self.buttons.items():
            btn = tk.Button(window, text=name, width=20, height=2, command=action)
            btn.pack()
            self.button_widgets[name] = btn
        self.button_widgets["スナップショット"].config(state='disabled')
        self.button_widgets["OCR"].config(state='disabled')
        #
        # for button_text, button_command in self.buttons.items():
        #     tk.Button(window, text=button_text, width=20, command=button_command).pack()  # ボタンの作成と配置

        self.canvas = tk.Canvas(self.window, width=640, height=480)  # 画像を表示するためのキャンバスの作成
        self.canvas.pack()  # キャンバスをウィンドウに配置

        self.delay = 50  # 更新の遅延時間の設定
        self.update()  # 初回の更新呼び出し
        self.ocr_label = {}

        self.tagger = MeCab.Tagger("-Owakati")        self.vectorizer = TfidfVectorizer()
        self.bert_cache = {}
        self.tfidf = None

        self.previous_img_stats = None
        self.previous_clahe = None
        self.clahe_threshold = 1.0

    def open_camera(self):
        try:
            if not self.camera_open:  # カメラがまだオープンしていない場合
                self.vid = cv2.VideoCapture(self.video_source)  # ビデオキャプチャオブジェクトの作成

                self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 4)
                self.vid.set(cv2.CAP_PROP_FPS, 30)

                if self.vid.isOpened():  # ビデオキャプチャが成功した場合
                    self.camera_open = True  # カメラがオープンした状態に設定
                    self.canvas.config(width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                       height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # キャンバスのサイズをビデオフレームのサイズに設定
                    self.button_widgets["スナップショット"].config(state='normal')
        except Exception as e:
            print(f"カメラを開くときにエラーが発生しました: {e}")
            self.camera_open = False

    def take_snapshot(self):
        if self.camera_open:  # カメラがオープンしている場合
            ret, frame = self.vid.read()  # フレームの読み取り
            if ret:  # フレームが正常に読み取られた場合
                self.img_bgr = frame.view()  # BGR画像のコピー
                self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.img_rgb))  # PIL画像からTkinterのPhotoImageに変換
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # キャンバスに画像を表示

                plt.figure("Snapshot")  # 新しい図を作成
                plt.imshow(self.img_rgb)  # RGB画像を表示
                plt.axis('off')  # 軸を非表示に設定
                plt.show()  # 画像を表示
                self.button_widgets["OCR"].config(state='normal')

    def perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):  # カメラがオープンしており、画像が存在する場合
            self.thread_pool.submit(self._perform_ocr)  # OCR処理をスレッドプールで実行
            self.button_widgets["スナップショット"].config(state='disabled')

    def _perform_ocr(self):
        if self.camera_open and hasattr(self, 'img_bgr'):  # カメラがオープンしており、画像が存在する場合
            self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)  # BGR画像をグレースケールに変換
            cv2.imwrite('gray_paddle.jpg', self.img_gray)  # グレースケール画像をファイルに保存
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHEオブジェクトの作成
            # cl1 = clahe.apply(self.img_gray)  # CLAHEの適用

            roi_image = analyze_roi(self.img_gray)

            clahe_image = adjust_clahe_params(self, roi_image)
            blurred = automatic_gaussian_blur(clahe_image)  # ガウスぼかしの適用

            kernel = np.ones((3, 3), dtype=np.uint8)  # カーネルの作成
            dilated = cv2.dilate(blurred, kernel, iterations=2)  # 膨張処理の適用
            eroded = cv2.erode(dilated, kernel, iterations=1)  # 収縮処理の適用

            canny_edges = cv2.Canny(eroded, 30, 150)  # Cannyエッジ検出の適用
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # Sobelフィルタ（x方向）の適用
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Sobelフィルタ（y方向）の適用
            sobel_edges = cv2.magnitude(sobel_x, sobel_y)  # Sobelエッジの計算
            sobel_edges = cv2.convertScaleAbs(sobel_edges)  # 絶対値の変換

            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian_edges = cv2.convertScaleAbs(laplacian)

            edged = np.maximum(canny_edges, sobel_edges)  # CannyエッジとSobelエッジの最大値を計算
            edged = np.maximum(edged, laplacian_edges)

            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭の検出
            image_with_contours = cv2.drawContours(self.img_gray.copy(), contours, -1, (0, 255, 0), 2)  # 輪郭を画像に描画

            scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # スケールのリスト
            all_results = []  # OCR結果を格納するリスト

            for scale in scales:  # 各スケールに対して処理
                scaled_img = cv2.resize(image_with_contours.copy(), None, fx=scale, fy=scale)  # 画像のリサイズ
                if scaled_img is not None:

                    result_en = self.ocr_en.ocr(scaled_img)  # OCR処理の実行

                    if result_en is not None:  # OCR結果が存在する場合
                        for line in result_en:  # 各ラインに対して処理
                            if line is not None:
                                for word_info in line:  # 各単語に対して処理
                                    # word = word_info[1][0]  # 単語の取得
                                    word = preprocess_japanese_text(word_info[1][0])
                                    similarity_score = 0
                                    calculate_word = None
                                    result = self.combined_similarity(word)  # 単語の類似度計算
                                    if result is not None:
                                        similarity_score, calculate_word = result
                                    confidence = word_info[1][1]  # OCRの信頼度
                                    if confidence >= 0.85 and similarity_score is not None and similarity_score >= 0.90:  # 信頼度と類似度が閾値を超えた場合
                                        rect_color = (152, 255, 152)  # 長方形の色
                                        coordinates = np.array(word_info[0]) / float(scale)  # 単語の座標
                                        x_min = coordinates[:, 0].min()  # 最小x座標
                                        y_min = coordinates[:, 1].min()  # 最小y座標
                                        x_max = coordinates[:, 0].max()  # 最大x座標
                                        y_max = coordinates[:, 1].max()  # 最大y座標

                                        word = calculate_word
                                        all_results.append(
                                            (word, [x_min, y_min, x_max, y_max], confidence, rect_color))  # 結果をリストに追加

                                    # else:
                                    #     rect_color = (0, 0, 255)
                                    #     coordinates = word_info[0]
                                    #     x_min = float(min(pt[0] for pt in coordinates)) / scale
                                    #     y_min = float(min(pt[1] for pt in coordinates)) / scale
                                    #     x_max = float(max(pt[0] for pt in coordinates)) / scale
                                    #     y_max = float(max(pt[1] for pt in coordinates)) / scale
                                    #     all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))

            all_results.sort(key=lambda x: -x[2])  # 信頼度でソート

            keep = [1] * len(all_results)  # 結果を保持するためのリスト
            for i in range(len(all_results)):  # 非最大抑制の適用
                if keep[i] == 0:
                    continue
                _, box1, _, _ = all_results[i]
                for j in range(i + 1, len(all_results)):
                    if keep[j] == 0:
                        continue
                    _, box2, _, _ = all_results[j]
                    if calculate_image_iou(box1, box2) > 0.3:  # IOUが閾値を超えた場合
                        keep[j] = 0  # 結果を保持しない

            result_img = self.img_bgr.copy()  # 結果画像のコピー

            for k, (word, box, _, rect_color) in enumerate(all_results):  # 結果を画像に描画
                if keep[k] == 1:
                    x_min, y_min, x_max, y_max = map(int, box)  # 座標を整数に変換
                    draw_transparent_rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color,
                                               alpha=0.5)  # 長方形を描画

                    # cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color, thickness=2)  # 長方形を描画
                    result_img = draw_text(result_img, word, (x_min, y_min - 5))  # テキストを描画

            detected_objects = any(keep)

            height, width, _ = result_img.shape
            if detected_objects:

                center = (width - 50, height - 50)
                radius = 40
                cv2.circle(result_img, center, radius, (0, 255, 0), thickness=3)

                play_sound("japanese_audio.mp3")
            else:

                pt1 = (width - 60, height - 60)
                pt2 = (width - 20, height - 20)
                cv2.line(result_img, pt1, pt2, (0, 0, 255), thickness=3)
                pt1 = (width - 20, height - 60)
                pt2 = (width - 60, height - 20)
                cv2.line(result_img, pt1, pt2, (0, 0, 255), thickness=3)

            cv2.imwrite('Result_paddle.jpg', result_img)  # 結果画像をファイルに保存

            import matplotlib.pyplot as plt  # matplotlibのインポート

            plt.figure("Result")  # 新しい図を作成
            if result_img is None:  # 結果画像が存在しない場合
                print("The result image is None, no image to display")
            else:  # 画像が存在する場合
                rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # RGBに変換
                print("Type of the image array: ", type(rgb_img))  # Image array type
                print("Shape of the image array: ", rgb_img.shape)  # Image array shape
                try:  # Try to display image
                    plt.imshow(rgb_img)  # 画像の表示
                except Exception as e:  # Exception handling if imshow fails
                    print(f"An error occurred when displaying the image: {e}")
            plt.axis('off')  # 軸を非表示に設定
            plt.show()  # 画像の表示
            self.button_widgets["スナップショット"].config(state='normal')
            self.button_widgets["OCR"].config(state='disabled')

            # ocr_result_en = '\n'.join([word_info[1][0] for line in result_en for word_info in line])  # OCR結果の連結（コメントアウトされている）
            # self.result_label.config(text=ocr_result_en)  # OCR結果をラベルに表示（コメントアウトされている）

    def close(self):
        if self.vid and self.vid.isOpened():  # ビデオキャプチャがオープンしている場合
            self.vid.release()  # ビデオキャプチャを解放
        cv2.destroyAllWindows()  # OpenCVの全ウィンドウを閉じる
        self.thread_pool.shutdown(wait=True)
        self.window.quit()  # Tkinterウィンドウを終了

    def combined_similarity(self, word):
        for accepted_word, possible_words in self.ocr_label.items():  # 受け入れ可能な単語リストを確認
            if word in possible_words:
                word = accepted_word
                return 1.0, word  # 単語が受け入れ可能な単語リストに存在する場合、類似度は1.0

        max_sim = 0  # 最大類似度の初期化
        best_match = word
        for accepted_word in self.ocr_label.keys():  # すべての受け入れ可能な単語と比較

            token1 = nlp(word)  # 単語をSpaCyトークンに変換
            token2 = nlp(accepted_word)  # 受け入れ可能な単語をSpaCyトークンに変換
            spacy_sim = token1.similarity(token2)  # SpaCyによる類似度計算
            fuzzy_sim = fuzz.ratio(word.lower(), accepted_word.lower()) / 100.0  # FuzzyWuzzyによる類似度計算

            self.prepare_texts(word, accepted_word)
            # bert_sim = self.get_bert_similarity(word, accepted_word)
            jaccard_sim = self.jaccard_similarity(word, accepted_word)
            levenshtein_sim = self.levenshtein_similarity(word, accepted_word)

            sim = 0.15 * spacy_sim + 0.15 * fuzzy_sim + 0.35 * jaccard_sim + 0.35 * levenshtein_sim

            if sim > max_sim:  # 最大類似度を更新
                max_sim = sim
                best_match = accepted_word

        return max_sim, best_match  # 最大類似度を返す

        #     token1 = nlp(word)  # 単語をSpaCyトークンに変換
        #     token2 = nlp(accepted_word)  # 受け入れ可能な単語をSpaCyトークンに変換
        #     spacy_sim = token1.similarity(token2)  # SpaCyによる類似度計算
        #
        #     fuzzy_sim = fuzz.ratio(word.lower(), accepted_word.lower()) / 100.0  # FuzzyWuzzyによる類似度計算
        #     sim = max(spacy_sim, fuzzy_sim)  # 最大類似度を選択
        #
        #     if sim > max_sim:  # 最大類似度を更新
        #         max_sim = sim
        # return max_sim  # 最大類似度を返す

    def process_ocr(self, image_with_contours):
        result_en = None
        try:
            if image_with_contours is not None:
                result_en = self.ocr_en.ocr(image_with_contours)  # OCR処理的执行
        except IndexError:
            print("An IndexError occurred. Continuing with program execution...")
        except Exception as e:
            print(f"An error occurred: {str(e)}. Continuing with program execution...")
        finally:
            return result_en

    def update(self):
        if self.camera_open:  # カメラがオープンしている場合
            ret, frame = self.vid.read()  # フレームの読み取り
            if ret:  # フレームが正常に読み取られた場合
                img_bgr = frame.view()
                gray = cv2.cvtColor(frame.view(), cv2.COLOR_BGR2GRAY)  # BGR画像をグレースケールに変換

                # gray = correct_lens_distortion(gray, np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]]),
                #                               np.array([0, 0, 0, 0]))

                roi_image = analyze_roi(gray)
                clahe_image = adjust_clahe_params(self, roi_image)
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHEオブジェクトの作成
                # cl1 = clahe.apply(gray)  # CLAHEの適用
                blurred = automatic_gaussian_blur(clahe_image)  # ガウスぼかしの適用

                kernel = np.ones((3, 3), dtype=np.uint8)  # カーネルの作成
                dilated = cv2.dilate(blurred, kernel, iterations=2)  # 膨張処理の適用
                eroded = cv2.erode(dilated, kernel, iterations=1)  # 衰退処理の適用

                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(apply_canny, eroded): 'canny',
                        executor.submit(apply_sobel, blurred): 'sobel',
                        executor.submit(apply_laplacian, blurred): 'laplacian'
                    }

                    results = {}
                    for future in futures:
                        result_name = futures[future]
                        results[result_name] = future.result()

                canny_edges = results['canny']
                sobel_edges = results['sobel']
                laplacian_edges = results['laplacian']

                edged = np.maximum(canny_edges, sobel_edges)
                edged = np.maximum(edged, laplacian_edges)

                contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭の検出
                image_with_contours = cv2.drawContours(gray.copy(), contours, -1, (0, 255, 0), 2)  # 輪郭を画像に描画

                try:
                    future = self.thread_pool.submit(self.process_ocr, image_with_contours)  # OCR処理をスレッドプールで実行
                    result_en = future.result()  # 結果の取得

                except Exception as e:
                    print(f"OCR process failed with error: {e}")
                    result_en = None  # or set a reasonable default value

                result_img = img_bgr.copy()  # 結果画像のコピー

                if result_en is not None:  # OCR results exist
                    # Filter and sort the OCR results
                    filtered_words = []
                    for line in result_en:
                        if line:
                            # Filter words with confidence >= 0.80
                            filtered_words.extend([
                                word_info for word_info in line
                                if word_info[1][1] >= 0.80
                            ])

                    # Sort by confidence (descending)
                    filtered_words.sort(key=lambda wi: wi[1][1], reverse=True)

                    all_results = []  # OCR結果を格納するリスト

                    for word_info in filtered_words:  # 各単語に対して処理
                        word = preprocess_japanese_text(word_info[1][0])
                        # word = word_info[1][0]  # 単語の取得
                        print(word)
                        similarity_score = 0
                        calculate_word = None
                        result = self.combined_similarity(word)  # 単語の類似度計算
                        if result is not None:
                            similarity_score, calculate_word = result

                        confidence = word_info[1][1]  # OCRの信頼度
                        if similarity_score is not None and similarity_score >= 0.85:  # 信頼度と類似度が閾値を超えた場合
                            coordinates = np.array(word_info[0])  # 単語の座標
                            x_min = int(coordinates[:, 0].min())  # 最小x座標
                            y_min = int(coordinates[:, 1].min())  # 最小y座標
                            x_max = int(coordinates[:, 0].max())  # 最大x座標
                            y_max = int(coordinates[:, 1].max())  # 最大y座標
                            rect_color = (152, 255, 152)  # 長方形の色
                            word = calculate_word
                            all_results.append(
                                (word, [x_min, y_min, x_max, y_max], confidence, rect_color))  # 結果をリストに追加
                        # else:
                        #     rect_color = (0, 0, 255)
                        #     coordinates = word_info[0]
                        #     x_min = int(min(pt[0] for pt in coordinates))
                        #     y_min = int(min(pt[1] for pt in coordinates))
                        #     x_max = int(max(pt[0] for pt in coordinates))
                        #     y_max = int(max(pt[1] for pt in coordinates))
                        #     all_results.append((word, [x_min, y_min, x_max, y_max], confidence, rect_color))

                    # Apply Non-Maximum Suppression (NMS) based on IOU
                    keep = [1] * len(all_results)  # 結果を保持するためのリスト
                    for i in range(len(all_results)):  # 非最大抑制の適用
                        if keep[i] == 0:
                            continue
                        _, box1, _, _ = all_results[i]
                        for j in range(i + 1, len(all_results)):
                            if keep[j] == 0:
                                continue
                            _, box2, _, _ = all_results[j]
                            if calculate_video_ciou(box1, box2) >= 0.40:  # IOUが閾値を超えた場合
                                keep[j] = 0  # 結果を保持しない

                    for k, (word, box, _, rect_color) in enumerate(all_results):  # 結果を画像に描画
                        if keep[k] == 1:
                            x_min, y_min, x_max, y_max = box  # 座標を取得
                            draw_transparent_rectangle(result_img, (x_min, y_min), (x_max, y_max), rect_color,
                                                       alpha=0.5)  # 長方形を描画
                            result_img = draw_text(result_img, word, (x_min, y_min - 10))  # テキストを描画

                    detected_objects = any(keep)

                    height, width, _ = result_img.shape
                    center = (width - 80, height - 160)
                    if detected_objects:

                        radius = 40
                        cv2.circle(result_img, center, radius, (0, 255, 0), thickness=3)

                        # play_sound("japanese_audio.mp3")
                    else:

                        size = 40
                        cv2.line(result_img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size),
                                 (0, 0, 255),
                                 thickness=3)
                        cv2.line(result_img, (center[0] + size, center[1] - size), (center[0] - size, center[1] + size),
                                 (0, 0, 255),
                                 thickness=3)

                    self.photo = ImageTk.PhotoImage(
                        image=Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))  # 画像をTkinter形式に変換
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # Tkinterキャンバスに画像を描画
                    # self.result_label.config(text=ocr_result_en)  # OCR結果をラベルに表示（コメントアウトされている）

        self.window.after(self.delay, self.update)  # 指定した遅延時間後にupdateメソッドを再実行

    def update_ocr_label(self, ocr_label):
        for corrected, possible_responses in ocr_to_accepted_words.items():
            if ocr_label == corrected:
                self.ocr_label = {corrected: possible_responses}

    def prepare_texts(self, text1, text2):
        if text1.strip() and text2.strip():
            self.vectorizer = TfidfVectorizer(stop_words=None)
            try:
                self.tfidf = self.vectorizer.fit_transform([text1, text2])
            except ValueError:
                print(f"Both text1: '{text1}' and text2: '{text2}' may be empty or contain only stop words.")

    def get_bert_embedding(self, text):
        if text in self.bert_cache:
            return self.bert_cache[text]

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        self.bert_cache[text] = embedding
        return embedding

    @lru_cache(maxsize=100)

    def tokenize_japanese(self, text):
        text = re.sub(r'\W', ' ', text)
        return set(self.tagger.parse(text).strip().split())

    def jaccard_similarity(self, text1, text2):
        set1 = self.tokenize_japanese(text1)
        set2 = self.tokenize_japanese(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def levenshtein_similarity(self, text1, text2):
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        return 1 - (distance / max_len)


if __name__ == "__main__":
    try:
        root = tk.Tk()  # Tkinterウィンドウの作成
        app = App(root, "Tkinter and OpenCV")  # アプリケーションのインスタンス化
        root.mainloop()  # Tkinterのメインループを開始
    except KeyboardInterrupt:  # キーボード割り込みが発生した場合
        if app.vid is not None and app.vid.isOpened():  # ビデオキャプチャがオープンしている場合
            app.vid.release()  # ビデオキャプチャを解放
            cv2.destroyAllWindows()  # OpenCVの全ウィンドウを閉じる
            root.quit()  # Tkinterウィンドウを終了
