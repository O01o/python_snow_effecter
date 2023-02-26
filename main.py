# ステップ1. インポート
import PySimpleGUI as sg  # PySimpleGUIをsgという名前でインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import numpy as np  # numpyのインポート
import cv2  # OpenCV（python版）のインポート
import dlib  #Dlibのインクルード
# ---- ここからGUIの設定込みの顔画像認識 ----
# ステップ2. デザインテーマの設定
sg.theme('DarkTeal7')

from pixel_warping_effect import *
# from test import *

display_size = (400, 300)

def scale_to_height(img, height):
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))
    return dst
# ステップ3. ウィンドウの部品とレイアウト
layout = [
	[sg.Text('認識対象の画像ファイルを指定してください')],
	[sg.Text('入力画像', size=(10, 1)), sg.Input(), sg.FileBrowse('ファイルを選択', key='inputFilePath'), sg.Button('読み込み', key='read')],
	[sg.Text('出力画像', size=(10, 1)), sg.Input('output.jpg', key='outputFile'), sg.FileBrowse('ファイルを選択', key='outputFilePath')],
	[sg.Button('認識開始', key='run'), sg.Button('画像保存', key='save'), sg.Button('終了', key='exit')],
    [sg.Image(filename=f'/Users/a010/src/public/img-processing/face_recognition/no_image.gif', size=display_size, key='-input_image-')],
    [sg.Output(size=(80,10))]
]

# ステップ4. ウィンドウの生成
window = sg.Window('画像から顔を認識するツール', layout, location=(400, 20))

# ステップ5. イベントループ
while True:
	event, values = window.read()
	
	if event in (None, 'exit'): #ウィンドウのXボタンまたは”終了”ボタンを押したときの処理
		break
	
	if event == 'read': #「読み込み」ボタンが押されたときの処理
		# 画像の読み込み処理
		print("Read image = " + values['inputFilePath'])
		orig_img = cv2.imread(values['inputFilePath'])  # OpenCVの画像読み込み関数を利用
		# 画像が大きいのでサイズを1/2にする．shapeに画像のサイズ(row, column)が入っている
		height, width, color = orig_img.shape  #shape[0]は行数（画像の縦幅）
		orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))  #OpenCVのresize関数
		disp_img = scale_to_height(orig_img, display_size[1])
		imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
		window['-input_image-'].update(data=imgbytes)

	if event == 'run':  #「認識開始」ボタンが押されて時の処理
		print("Start recognition")
		# ---- 顔認識エンジンセット ----
		detector = dlib.get_frontal_face_detector()  #Dlibに用意されている検出器をセット
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 学習済みファイル読み込み
		img = orig_img.copy()  #処理結果表示用の変数 img を用意して，orig_imgをコピー
		print("loaded img shape: (height, width, color) ->", orig_img.shape)
		faces = detector(orig_img[:, :, ::-1])  #画像中の全てを探索して「顔らしい箇所」を検出
		if len(faces) > 0:  # 顔を見つけたら以下を処理する
			for face in faces:  #全ての顔 faces から一つの face を取り出して
				parts = predictor(orig_img, face).parts()  #顔パーツ推定
				count = 0
				left_cheek = 2
				right_cheek = 14
				jaw = 8
				for i in parts:
					# if count == left_cheek or count == jaw or count == right_cheek:cv2.circle(img, (i.x, i.y), 1, (0, 255, 0), -1) # 点をプロット
					if count == left_cheek:
						img = pixel_warping_effect(img, center=(i.y, i.x), size=80, rad=60)
					elif count == right_cheek:
						img = pixel_warping_effect(img, center=(i.y, i.x), size=80, rad=300)
					elif count == jaw:
						img = pixel_warping_effect(img, center=(i.y, i.x), size=80, rad=0)
					count += 1
					# print(i.x, i.y)
				print("顔のパーツの個数:", len(parts))

		outimg = img
		print("Finish recognition")

	if event == 'save': #「画像保存」ボタンが押されたときの処理
		print("Write image -> " + values['outputFile'])
		cv2.imwrite(values['outputFile'], outimg)  # OpenCVの画像書き出し関数を利用
		orig_img = cv2.imread(values['outputFile'])  # OpenCVの画像読み込み関数を利用
		# 画像が大きいのでサイズを1/2にする．shapeに画像のサイズ(row, column)が入っている
		height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
		width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）
		orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))  #OpenCVのresize関数
		disp_img = scale_to_height(orig_img, display_size[1])
		imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
		window['-input_image-'].update(data=imgbytes)
	
window.close()
