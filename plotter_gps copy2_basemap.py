import numpy as np
from multiprocessing import shared_memory
import config as CON

import pyautogui as pag
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from multiprocessing import Lock
from config_manager import read_config
import scipy.interpolate as interpolate

# グローバル変数としてロックを保持
global_lock = Lock()

def start_plotting_one_figure(shared_mem_name_GPS):
    config = read_config()
    USEFLAG_GPS = int(config['DEFAULT']['USEFLAG_GPS'])
    FIGFLAG     = int(config['DEFAULT']['FIGFLAG'])

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
        existing_shm_GPS = shared_memory.SharedMemory(name=shared_mem_name_GPS)
        array_GPS = np.ndarray(CON.GPS_SHAPE, dtype=CON.GPS_DTYPE, buffer=existing_shm_GPS.buf)

        # Basemapを使った地図プロット
        ax12 = fig.add_subplot(gs[6:10, 1])
        m = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax12)

        # 地図の描画
        m.drawcoastlines()
        m.drawcountries()
        m.drawrivers(color="blue")
        # m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])  # 緯度目盛り
        # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])  # 経度目盛り

        ax12.set_xlabel("Longitude", fontsize=12)
        ax12.set_ylabel("Latitude", fontsize=12)
        ax12.set_title("GPS Data on Map", fontsize=14)

        # GPSデータ用のプロットオブジェクト
        line12, = ax12.plot([], [], 'ko', markersize=5)  # 初期状態は空

    plt.ion()  # インタラクティブモードをON

    while True:
        with global_lock:
            index_gps = np.argmax(array_GPS[:, 0]) if USEFLAG_GPS else None

            if index_gps and USEFLAG_GPS:
                if index_gps > 10:
                    gps_indices = np.where(array_GPS[:, 0] == 0)[0]
                    if len(gps_indices) > 0:
                        gps_index = min(gps_indices)
                        array_gps_ExcludeZero = array_GPS[:gps_index]
                    else:
                        gps_index = len(array_GPS) - 1
                        array_gps_ExcludeZero = array_GPS[:gps_index]

                    # 時間データのリサンプリング
                    time_data = array_gps_ExcludeZero[:, 2]
                    new_time_data = np.round(np.arange(time_data[0], time_data[-1], 0.1), 8)
                    interp_func = interpolate.interp1d(
                        time_data, array_gps_ExcludeZero, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate"
                    )
                    array_gps_ExcludeZero_interp = interp_func(new_time_data)
                    array_gps_ExcludeZero_interp[:, 2] -= array_gps_ExcludeZero_interp[0, 2]

                    lon = array_gps_ExcludeZero_interp[:, 4]  # longitude
                    lat = array_gps_ExcludeZero_interp[:, 3]  # latitude

                    # 緯度・経度データをプロット
                    line12.set_xdata(lon)
                    line12.set_ydata(lat)

                    # 表示範囲の調整
                    # ax12.set_xlim(min(lon)-0.01, max(lon)+0.01)
                    # ax12.set_ylim(min(lat)-0.01, max(lat)+0.01)
                    ax12.relim()
                    ax12.autoscale_view()
                    
            fig.tight_layout()
            plt.draw()
            plt.pause(1)  # 1秒ごとに更新
