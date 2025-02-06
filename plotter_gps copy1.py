# plotter.py

import numpy as np
from multiprocessing import shared_memory
import config as CON

# add mtsuzuki 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))

import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter

import pyautogui as pag

import folium
import geocoder

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize

import seaborn as sns

from multiprocessing import Lock
from config_manager import read_config



# グローバル変数としてロックを保持
global_lock = Lock()

def start_plotting_one_figure(shared_mem_name_GPS):

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

        ax12 = fig.add_subplot(gs[6:10, 1])

        line12, = ax12.plot([], [], 'k-')
        ax12.set_ylabel("latitude")
        ax12.set_xlabel("longitude")

        # カラーマップを生成
        cmap = cm.get_cmap("jet")
        norm = Normalize(vmin=0, vmax=20)

        # カラーバーを追加するためのフラグ
        colorbar_added = False
        colorbar = None

        tmpUpdateCounter_LFHF_GPS = 0

    plt.ion()

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

            plt.draw()
            plt.pause(0.05)

            fig.tight_layout()

def get_current_location():
    # 現在地を取得
    g = geocoder.ip('me')
    return g.latlng

def update_map(latitudes, longitudes, map_file='map.html', initial_location=None):
    # 地図の初期設定
    if initial_location is None:
        initial_location = [35.6895, 139.6917]  # デフォルトは東京
    m = folium.Map(location=initial_location, zoom_start=12)
    
    # 経路を描画
    coordinates = list(zip(latitudes, longitudes))
    folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(m)
    
    # 地図の範囲を設定
    m.fit_bounds([coordinates[0], coordinates[-1]])
    
    # 地図を保存
    m.save(map_file)