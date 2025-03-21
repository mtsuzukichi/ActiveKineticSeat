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

import serial,sys,time


# フラグ判定モード（"circle"=円判定, "line"=ライン判定）
# flag_mode = "circle"
flag_mode = "line"

# circle_origin = (35.0549, 137.1631)  # 本社T/C入口
# circle_origin = (35.0532, 137.1653)  # 東バンク入口
circle_origin = (3.495835912666666611e+01,1.371432051900000033e+02)  # 自宅付近 テスト用
circle_radius_m = 3  # 円の半径をメートル単位で指定

# ラインの定義（2点で決まる）
# line_point1 = (35.0531, 137.1653)  # ライン 東バンク１
# line_point2 = (35.0532, 137.1655)  # ライン 東バンク２
# line_point1 = (35.0550, 137.1609)  # ライン 本社 構内路 オーバーブリッジ手前
# line_point2 = (35.0550, 137.1611)  # ライン 本社 構内路 オーバーブリッジ手前
line_point1 = (35.222120, 138.903772)  # ライン 東富士 第二水直 南側
line_point2 = (35.222156, 138.903682)  # ライン 東富士 第二水直 南側


line_segments = [
    # ((34.958242, 137.143170), (34.958405, 137.143268)),  # 自宅 テスト用ライン１
    # ((34.958263, 137.143110), (34.958422, 137.143223)),  # 自宅 テスト用ライン２
    # ((34.958177, 137.143282), (34.958190, 137.143408)),  # 自宅 テスト用ライン１
    # ((34.958229, 137.143353), (34.958422, 137.143223)),  # 自宅 テスト用ライン２
    # ((35.222120, 138.903772), (35.222156, 138.903682)),  # 東富士 第二水直 南側
    # ((35.228230, 138.907711), (35.228202, 138.907791)),  # 東富士 第二水直 北側
    # ((35.227363, 138.898499), (35.227162, 138.898976)),  # 東富士第1周回路 南バンク
    # ((35.229320, 138.907466), (35.229140, 138.907891)),  # 東富士第1周回路 南バンク
    # ((35.231023, 138.889509), (35.231027, 138.889901)),  # 東富士第3周回路 内周路１
    # ((35.232067, 138.888676), (35.231789, 138.888647)),  # 東富士第3周回路 内周路２
    # ((35.232486, 138.887550), (35.232614, 138.887853)),  # 東富士第3周回路 内周路３
    # ((35.233201, 138.886413), (35.232819, 138.886490)),  # 東富士第3周回路 内周路４
    # ((35.234496, 138.885182), (35.234388, 138.885701)),  # 東富士第3周回路 内周路５
    # ((35.235484, 138.885094), (35.235492, 138.885678)),  # 東富士第3周回路 内周路６
    # ((35.233363, 138.888180), (35.233004, 138.888113)),  # 東富士第3周回路 内周路７

    # ((35.230841, 138.889401), (35.230813, 138.889936), 0),  # 東富士第3周回路 内周路１
    # ((35.232091, 138.888844), (35.231729, 138.888825), 0),  # 東富士第3周回路 内周路２
    # ((35.232522, 138.887481), (35.232571, 138.887966), 0),  # 東富士第3周回路 内周路３
    # ((35.233258, 138.886680), (35.232938, 138.886735), 0),  # 東富士第3周回路 内周路４
    # ((35.234392, 138.885085), (35.234292, 138.885730), 0),  # 東富士第3周回路 内周路５
    # ((35.235484, 138.885094), (35.235492, 138.885678), 0),  # 東富士第3周回路 内周路６
    # ((35.233363, 138.888180), (35.233004, 138.888113), 0),  # 東富士第3周回路 内周路７

    # E1 まわりのテスト用
    # ((35.225698, 138.906106), (35.225457, 138.906664), 0),  # 東富士 E1周り １
    # ((35.224649, 138.905526), (35.224530, 138.905797), 0),  # 東富士 E1周り ２
    # ((35.224213, 138.905272), (35.224139, 138.905513), 0),  # 東富士 E1周り ４
    # ((35.224327, 138.907800), (35.224208, 138.908054), 0.5),  # 東富士 E1周り ５
    # ((35.225436, 138.907942), (35.225472, 138.908142), 0),  # 東富士 E1周り ６

    # 中研さんRパーク 3/21最終版から２版のラインを延長
    ((35.169188, 137.053708), (35.169259, 137.053923), 0),  # 中研さん Rパーク １
    ((35.168922, 137.053700), (35.168952, 137.054036), 0),  # 中研さん Rパーク ２
    ((35.169166, 137.054234), (35.169210, 137.054400), 0),  # 中研さん Rパーク ３
    ((35.168816, 137.054288), (35.168877, 137.054492), 0),  # 中研さん Rパーク ４
   
    # 中研さんRパーク 3/21最終
    # ((35.169188, 137.053708), (35.169259, 137.053923), 0),  # 中研さん Rパーク １
    # ((35.168934, 137.053837), (35.168952, 137.054036), 0),  # 中研さん Rパーク ２
    # ((35.169166, 137.054234), (35.169210, 137.054400), 0),  # 中研さん Rパーク ３
    # ((35.168816, 137.054288), (35.168877, 137.054492), 0),  # 中研さん Rパーク ４

    # 必要に応じて追加
]


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
    print("distance_to_center:",distance_to_center)
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

def crosses_any_line(prev_lat, prev_lon, curr_lat, curr_lon, line_segments):
    """現在位置と前回の位置が指定された複数のラインのいずれかをまたいだか判定"""
    if prev_lat is None or prev_lon is None:
        return False, 0  # 初回は判定しない

    for line_point1, line_point2, delay in line_segments:
        if crosses_line(prev_lat, prev_lon, curr_lat, curr_lon, line_point1, line_point2):
            # return True  # どれかのラインをまたいでいたら即座に True を返す
            return True, delay  # どれかのラインをまたいでいたら即座に True を返す
    # return False  # どのラインもまたいでいない
    return False, 0  # どのラインもまたいでいない


def start_plotting_one_figure(shared_mem_name_GPS):

    # ハプティックデバイス
    ser = serial.Serial("COM18",921600)
    delay_HapOn = 0 # 単位：sec

    activate_flag = 0
    activate_flag_prev = 0
    prev_lat = None
    prev_lon = None
    current_time = 0
    flag_on_time = 0
    flag_HapOn = 0 

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
            # ax12.plot([line_point1[1], line_point2[1]], [line_point1[0], line_point2[0]], 'r-', linewidth=3, label="Threshold Line")
            for (point1, point2, _) in line_segments:
                ax12.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r-', linewidth=3, label="Threshold Line")

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
                        # activate_flag = crosses_line(prev_lat, prev_lon, current_lat, current_lon, line_point1, line_point2)
                        activate_flag, tmp_delay_HapOn = crosses_any_line(prev_lat, prev_lon, current_lat, current_lon, line_segments)

                        if activate_flag:
                            delay_HapOn = tmp_delay_HapOn

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


            if activate_flag == 1 and activate_flag_prev == 0:
                flag_on_time = current_time 
                print("current_time:",current_time)
                print("flag_on_time:",flag_on_time)
                flag_HapOn = 1


            # print("activate_flag_prev:",activate_flag_prev)
            # print("flag_on_time + delay_HapOn*1000",flag_on_time + delay_HapOn*1000)

            # if activate_flag_prev == 1 and current_time > flag_on_time + delay_HapOn*1000:
            if flag_HapOn == 1 and current_time > flag_on_time + delay_HapOn*1000:
                print("current_time:",current_time)
                print("flag_on_time:",flag_on_time)
                print("delay_HapOn",delay_HapOn)

                stime = 1.0
                ser.write(bytearray([ord('p'), 0,0b1111])) # 4つすべて
                time.sleep(stime)
                ser.write(bytearray([ord('p'), 0,0]))
                time.sleep(stime)

                activate_flag_prev = 0
                flag_HapOn= 0


            # activate_flag の状態を保存
            activate_flag_prev = activate_flag

