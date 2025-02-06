import numpy as np
from multiprocessing import shared_memory
import config as CON
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))

import scipy.interpolate as interpolate
import pyautogui as pag
import folium
import webbrowser
import tempfile
import time
from multiprocessing import Lock
from config_manager import read_config

# グローバル変数としてロックを保持
global_lock = Lock()

def start_plotting_one_figure(shared_mem_name_GPS):
    config = read_config()
    USEFLAG_GPS = int(config['DEFAULT']['USEFLAG_GPS'])
    FIGFLAG = int(config['DEFAULT']['FIGFLAG'])

    screen_width, screen_height = pag.size()
    window_width = screen_width // 2
    window_height = screen_height // 10 * 8
    window_pos_x = 0
    window_pos_y = screen_height // 10 * 1

    if USEFLAG_GPS:
        existing_shm_GPS = shared_memory.SharedMemory(name=shared_mem_name_GPS)
        array_GPS = np.ndarray(CON.GPS_SHAPE, dtype=CON.GPS_DTYPE, buffer=existing_shm_GPS.buf)

    while True:
        with global_lock:
            index_gps = np.argmax(array_GPS[:, 0]) if USEFLAG_GPS else None
            print(index_gps)
            if index_gps and USEFLAG_GPS and index_gps > 0:
                gps_indices = np.where(array_GPS[:, 0] == 0)[0]  # 余分な行を削除
                if len(gps_indices) > 0:
                    gps_index = min(gps_indices)
                    array_gps_ExcludeZero = array_GPS[:gps_index]
                else:
                    gps_index = len(array_GPS) - 1  # 0が見つからない場合は全ての行を使用
                    array_gps_ExcludeZero = array_GPS[:gps_index]

                # 時間軸をリサンプリング
                time_data = array_gps_ExcludeZero[:, 2]
                new_time_data = np.round(np.arange(time_data[0], time_data[-1], 0.1), 8)
                interp_func = interpolate.interp1d(time_data, array_gps_ExcludeZero, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
                array_gps_ExcludeZero_interp = interp_func(new_time_data)
                array_gps_ExcludeZero_interp[:, 2] = array_gps_ExcludeZero_interp[:, 2] - array_gps_ExcludeZero_interp[0, 2]

                lon = array_gps_ExcludeZero_interp[:, 4]  # longitude
                lat = array_gps_ExcludeZero_interp[:, 3]  # latitude

                # 地図の作成
                initial_location = [lat[0], lon[0]]
                m = folium.Map(location=initial_location, zoom_start=12)
                coordinates = list(zip(lat, lon))
                folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(m)
                m.fit_bounds([coordinates[0], coordinates[-1]])

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
                    m.save(temp_file.name)
                    output_file = temp_file.name
                
                # デフォルトのブラウザで地図を開く
                webbrowser.open(f"file://{output_file}")

                # 一時ファイルを削除（オプション）
                time.sleep(5)  # 5秒待機してから削除
                os.remove(output_file)

            time.sleep(1)  # 更新間隔（例：1秒）

