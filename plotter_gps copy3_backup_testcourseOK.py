# plotter.py

import numpy as np
from multiprocessing import shared_memory
import config as CON

# add mtsuzuki 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))

import scipy.interpolate as interpolate

import pyautogui as pag

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import seaborn as sns

from multiprocessing import Lock
from config_manager import read_config
from geopy.distance import geodesic

from adjustText import adjust_text

# フラグ判定モード（"circle"=円判定, "line"=ライン判定）
flag_mode = "circle"
# flag_mode = "line"

# circle_origin = (35.0549, 137.1631)  # 本社T/C入口
circle_origin = (35.0532, 137.1653)  # 東バンク入口
# circle_origin = (3.495835912666666611e+01,1.371432051900000033e+02)  # 自宅付近 テスト用
circle_radius_m = 1  # 円の半径をメートル単位で指定

# ラインの定義（2点で決まる）
line_point1 = (35.0531, 137.1653)  # ライン 東バンク１
line_point2 = (35.0532, 137.1655)  # ライン 東バンク２

# 直前の位置を保持（初回はNone）
prev_lat = None
prev_lon = None


# グローバル変数としてロックを保持
global_lock = Lock()

def meters_to_degrees(meters, lat):
    """ メートルを緯度・経度の度数に変換（緯度に依存） """
    lat_degree = meters / 111139  # 1度 ≈ 111,139m
    lon_degree = meters / (111139 * np.cos(np.radians(lat)))  # 経度は緯度で補正
    return lat_degree, lon_degree

# 関数: Haversine距離を計算
def haversine(lat1, lon1, lat2, lon2, unit="m"):
    """
    2点間のHaversine距離を計算する。
    
    Parameters:
        lat1, lon1 : float  -> 第1点の緯度・経度
        lat2, lon2 : float  -> 第2点の緯度・経度
        unit       : str    -> 距離の単位 ("m"=メートル, "km"=キロメートル, "mi"=マイル)

    Returns:
        距離（指定された単位）
    """
    R = 6371000  # 地球の半径（メートル）
    
    phi1, phi2 = np.radians([lat1, lat2])
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    distance_m = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # 距離単位を変換
    if unit == "km":
        return distance_m / 1000  # キロメートル
    elif unit == "mi":
        return distance_m * 0.000621371  # マイル
    return distance_m  # メートル


def check_position_in_circle(current_lat, current_lon, circle_lat, circle_lon, circle_radius_m, ax):
    """
    現在位置と円の中心の距離を計算し、円の内側にいるかを判定する。

    Parameters:
        current_lat (float): 現在の緯度
        current_lon (float): 現在の経度
        circle_lat (float): 円の中心の緯度
        circle_lon (float): 円の中心の経度
        circle_radius_m (float): 円の半径（メートル）
        ax (matplotlib.axes._subplots.AxesSubplot): プロットする軸

    Returns:
        int: 円の内側なら1、外側なら0
    """
    # 2点間の距離を計算
    distance_to_center = haversine(current_lat, current_lon, circle_lat, circle_lon)

    # 円の内側にいるかどうかを判定
    inside_flag = 1 if distance_to_center < circle_radius_m else 0

    # タイトルを更新
    status = "Inside" if inside_flag else "Outside"
    ax.set_title(f"Center Point: {circle_lat:.4f}, {circle_lon:.4f} ({status})")

    # 既存のテキストを削除
    for txt in ax.texts:
        txt.remove()

    # 距離を表示するテキストの初期位置
    text_x = circle_lon
    text_y = circle_lat + meters_to_degrees(circle_radius_m * 1.2, circle_lat)[0]  # 円の外側に配置

    # テキストを配置
    text = ax.text(text_x, text_y, 
                   f"{distance_to_center:.1f} m", 
                   fontsize=12, color="black", 
                   ha='center', va='bottom', 
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # `adjust_text` を使用してテキストの位置を自動調整
    adjust_text([text], ax=ax, expand_text=(1.2, 1.2), force_text=(0.1, 0.1))

    return inside_flag

def crosses_line(prev_lat, prev_lon, curr_lat, curr_lon, line_point1, line_point2):
    """現在位置と前回の位置が指定ラインをまたいだかどうかを判定"""
    if prev_lat is None or prev_lon is None:
        return False  # 初回は判定しない

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (prev_lon, prev_lat)
    B = (curr_lon, curr_lat)
    C = (line_point1[1], line_point1[0])
    D = (line_point2[1], line_point2[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)



def start_plotting_one_figure(shared_mem_name_GPS):

    prev_lat = None
    prev_lon = None

    config = read_config()
    USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])
    FIGFLAG	           = int(config['DEFAULT']['FIGFLAG'])

    sns.set_style("darkgrid")
    sns.set_palette("dark")

    screen_width, screen_height = pag.size()
    window_width  = screen_width // 2
    window_height = screen_height // 10 * 8
    window_pos_x  = 0
    window_pos_y  = screen_height // 10 * 1

    # 10行2列のグリッドスペックを作成
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(10, 2, figure=fig)
    fig.canvas.manager.window.wm_geometry(f"{window_width}x{window_height}+{window_pos_x}+{window_pos_y}")

    fig.suptitle("-RealTimeMonitor-", fontsize=16)


    if USEFLAG_GPS:
        # print("USEFLAG_GPS",USEFLAG_GPS)
        existing_shm_GPS = shared_memory.SharedMemory(name=shared_mem_name_GPS)
        array_GPS = np.ndarray(CON.GPS_SHAPE, dtype=CON.GPS_DTYPE, buffer=existing_shm_GPS.buf)

        ax12 = fig.add_subplot(gs[0:4, 0:2])

        line12, = ax12.plot([], [], 'k-')
        ax12.set_ylabel("latitude")
        ax12.set_xlabel("longitude")

        # メートル単位の半径を度数に変換
        lat_radius, lon_radius = meters_to_degrees(circle_radius_m, circle_origin[0])

        # 円をあらかじめ設定した中心に配置
        if flag_mode == "circle":
            circle = patches.Circle(circle_origin[::-1], lon_radius, color='blue', alpha=0.1, fill=True)
            ax12.add_patch(circle)
            ax12.scatter(circle_origin[1], circle_origin[0], color='blue', marker='o', s=100, label="Center Point")
        elif flag_mode == "line":
            ax12.plot([line_point1[1], line_point2[1]], [line_point1[0], line_point2[0]], 'r-', linewidth=3, label="Threshold Line")


        ax13 = fig.add_subplot(gs[5:10, 0:2])
        line13, = ax13.plot([], [], 'k-')
        ax13.set_ylabel("flag")
        ax13.set_xlabel("time[sec]")


    plt.ion()

    # フラグの履歴を格納するリスト
    flag_history = []
    time_history = []

    while True:
        with global_lock:
            index_gps   = np.argmax(array_GPS[:, 0]) if USEFLAG_GPS else None
            # print("index_gps",index_gps)

            if index_gps and USEFLAG_GPS:
                if index_gps > 10:
                # if index_gps > 10 and USEFLAG_GPS:
                    # print("USEFLAG_GPS",USEFLAG_GPS)

                    gps_indices = np.where(array_GPS[:, 0] == 0)[0]  # 余分な行を削除
                    if len(gps_indices) > 0:
                        gps_index = min(gps_indices)
                        array_gps_ExcludeZero = array_GPS[:gps_index]
                    else:
                        gps_index = len(array_GPS) - 1  # 0が見つからない場合は全ての行を使用
                        array_gps_ExcludeZero = array_GPS[:gps_index]
                        
                    # 時間軸が厳密に0.1秒刻みではないため、リサンプリングする
                    # 時間データ（3列目）
                    time_data = array_gps_ExcludeZero[:, 2]
                    # 0.1秒ごとの新しい時間軸を作成
                    new_time_data = np.round(np.arange(time_data[0], time_data[-1], 0.1),8)
                    # 線形補間関数を作成
                    interp_func = interpolate.interp1d(time_data, array_gps_ExcludeZero, axis=0, kind='linear',bounds_error=False, fill_value="extrapolate")

                    # 新しい時間軸に沿ってデータをリサンプリング
                    array_gps_ExcludeZero_interp = interp_func(new_time_data)

                    # 計測開始時刻を０秒にシフト（3列目：utc_time）
                    array_gps_ExcludeZero_interp[:,2] = array_gps_ExcludeZero_interp[:,2] - array_gps_ExcludeZero_interp[0,2]

                    lon = array_gps_ExcludeZero_interp[:, 4] # longitude
                    lat = array_gps_ExcludeZero_interp[:, 3] # latitude

                    line12.set_xdata(lon)
                    line12.set_ydata(lat)

                    ax12.relim()
                    ax12.autoscale_view()


                    # 現在地を取得
                    current_lat = array_gps_ExcludeZero_interp[-1, 3]
                    current_lon = array_gps_ExcludeZero_interp[-1, 4]
                    current_time = array_gps_ExcludeZero_interp[-1, 0]

                    if flag_mode == "circle":
                        activate_flag = check_position_in_circle(
                            current_lat, current_lon, circle_origin[0], circle_origin[1], circle_radius_m, ax12
                        )
                    elif flag_mode == "line":
                        activate_flag = crosses_line(prev_lat, prev_lon, current_lat, current_lon, line_point1, line_point2)

                    # activate_flag = check_position_in_circle(current_lat, current_lon, circle_origin[0], circle_origin[1], circle_radius_m, ax12)
                    print("activate_flag:",activate_flag)

                    time_history.append(current_time)
                    flag_history.append(activate_flag)

                    # print("current_time",current_time)
                    # print("time_history",time_history)

                    line13.set_xdata(np.array(time_history)/1000)
                    line13.set_ydata(np.array(flag_history))

                    ax13.relim()
                    ax13.autoscale_view()

                    # 位置を更新
                    prev_lat, prev_lon = current_lat, current_lon

            plt.draw()
            plt.pause(0.05)
            plt.show(block=False)

            fig.tight_layout()
