# plotter.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import config as CON

# add mtsuzuki 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))
from fnc_bandpass        import fnc_butter_bandpass
from fnc_ecg_normalize   import fnc_ecg_normalize
from fnc_remove_outlier  import fnc_remove_outlier_next_to_each_other
from fnc_Calc_LFHF_ccvTP import fnc_Calc_LFHF_ccvTP

import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter

import pyautogui as pag

import folium
import geocoder
from matplotlib import cm
import matplotlib.collections as mcoll
from scipy import interpolate
from scipy.interpolate import interp1d

import matplotlib
from matplotlib.colors import Normalize

from multiprocessing import Lock
from config_manager import read_config, write_config

import matplotlib.gridspec as gridspec


# グローバル変数としてロックを保持
global_lock = Lock()

"""バイオシグナル用プロット関数"""
def start_plotting_one_figure(shared_mem_name_Bio,shared_mem_name_Bio_LFHF,shared_mem_name_Bio_Peak,shared_mem_name_IMU,shared_mem_name_IMU_2,shared_mem_name_Wit,shared_mem_name_GPS):

    config = read_config()
    USEFLAG_BIO        = int(config['DEFAULT']['USEFLAG_BIO'])
    USEFLAG_IMU        = int(config['DEFAULT']['USEFLAG_IMU'])
    USEFLAG_IMU_2      = int(config['DEFAULT']['USEFLAG_IMU_2'])
    USEFLAG_WIT        = int(config['DEFAULT']['USEFLAG_WIT'])
    USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])
    USEFLAG_FTF        = int(config['DEFAULT']['USEFLAG_FTF'])
    USEFLAG_CAN        = int(config['DEFAULT']['USEFLAG_CAN'])
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


    if USEFLAG_BIO:
        existing_shm_Bio = shared_memory.SharedMemory(name=shared_mem_name_Bio)
        array_Bio        = np.ndarray(CON.BIO_SHAPE, dtype=CON.BIO_DTYPE, buffer=existing_shm_Bio.buf)

        existing_shm_Peak = shared_memory.SharedMemory(name=shared_mem_name_Bio_Peak)
        array_Peak        = np.ndarray(CON.BIO_SHAPE_PEAK, dtype=np.int32, buffer=existing_shm_Peak.buf)

        ax1 = fig.add_subplot(gs[0:2, 0])  # 1行目と2行目を結合
        ax2 = fig.add_subplot(gs[2:4, 0])
        ax3 = fig.add_subplot(gs[4:6, 0])
        ax4 = fig.add_subplot(gs[6:8, 0])
        ax5 = fig.add_subplot(gs[8:10, 0])

        line1, = ax1.plot([], [], 'k-')
        line2, = ax2.plot([], [], 'k-')
        line3, = ax3.plot([], [], 'k-')
        line4, = ax4.plot([], [], 'k-')
        line5, = ax5.plot([], [], 'k-')

        ax1.set_ylabel("ECG(-)")
        ax1.set_xlabel("Time(s)")

        tmpUpdateCounter = 0
        tmpSizePeakIdx   = 0
        
    if USEFLAG_IMU:
        existing_shm_IMU = shared_memory.SharedMemory(name=shared_mem_name_IMU)
        array_IMU = np.ndarray(CON.IMU_SHAPE, dtype=CON.IMU_DTYPE, buffer=existing_shm_IMU.buf)

        ax6  = fig.add_subplot(gs[0, 1])
        ax7  = fig.add_subplot(gs[1, 1])
        ax8  = fig.add_subplot(gs[2, 1])
        # ax9  = fig.add_subplot(gs[3, 1])
        # ax10 = fig.add_subplot(gs[4, 1])
        # ax11 = fig.add_subplot(gs[5, 1])

        line6,  = ax6.plot([], [], 'k-')
        line7,  = ax7.plot([], [], 'k-')
        line8,  = ax8.plot([], [], 'k-')
        # line9,  = ax9.plot([], [], 'k-')
        # line10, = ax10.plot([], [], 'k-')
        # line11, = ax11.plot([], [], 'k-')

        ax6.set_ylabel("accX(m/s^2)")
        # ax6.set_xlabel("Time(s)")

        ax7.set_ylabel("accY(m/s^2)")
        # ax7.set_xlabel("Time(s)")

        ax8.set_ylabel("accZ(m/s^2)")
        ax8.set_xlabel("Time(s)")
        
        # ax9.set_ylabel("gyroX(rad/s)")
        # # ax9.set_xlabel("Time(s)")
        
        # ax10.set_ylabel("gyroY(rad/s)")
        # # ax10.set_xlabel("Time(s)")
        
        # ax11.set_ylabel("gyroZ(rad/s)")
        # ax11.set_xlabel("Time(s)")
        
    if USEFLAG_IMU_2:
        existing_shm_IMU_2 = shared_memory.SharedMemory(name=shared_mem_name_IMU_2)
        array_IMU_2 = np.ndarray(CON.IMU_SHAPE, dtype=CON.IMU_DTYPE, buffer=existing_shm_IMU_2.buf)
        
        ax9  = fig.add_subplot(gs[3, 1])
        ax10 = fig.add_subplot(gs[4, 1])
        ax11 = fig.add_subplot(gs[5, 1])

        line9,  = ax9.plot([], [], 'k-')
        line10, = ax10.plot([], [], 'k-')
        line11, = ax11.plot([], [], 'k-')

        ax9.set_ylabel("accX(m/s^2)")
        # ax9.set_xlabel("Time(s)")
        
        ax10.set_ylabel("accY(m/s^2)")
        # ax10.set_xlabel("Time(s)")
        
        ax11.set_ylabel("accZ(m/s^2)")
        ax11.set_xlabel("Time(s)")
        

    if USEFLAG_WIT:
        existing_shm_Wit = shared_memory.SharedMemory(name=shared_mem_name_Wit)
        array_Wit = np.ndarray(CON.WIT_SHAPE, dtype=CON.WIT_DTYPE, buffer=existing_shm_Wit.buf)

        ax6  = fig.add_subplot(gs[0, 1])
        ax7  = fig.add_subplot(gs[1, 1])
        ax8  = fig.add_subplot(gs[2, 1])
        ax9  = fig.add_subplot(gs[3, 1])
        ax10 = fig.add_subplot(gs[4, 1])
        ax11 = fig.add_subplot(gs[5, 1])

        line6,  = ax6.plot([], [], 'k-')
        line7,  = ax7.plot([], [], 'k-')
        line8,  = ax8.plot([], [], 'k-')
        line9,  = ax9.plot([], [], 'k-')
        line10, = ax10.plot([], [], 'k-')
        line11, = ax11.plot([], [], 'k-')

        ax6.set_ylabel("accX(m/s^2)")
        ax6.set_xlabel("Time(s)")

        ax7.set_ylabel("accY(m/s^2)")
        ax7.set_xlabel("Time(s)")

        ax8.set_ylabel("accZ(m/s^2)")
        ax8.set_xlabel("Time(s)")
        
        ax9.set_ylabel("gyroX(rad/s)")
        ax9.set_xlabel("Time(s)")
        
        ax10.set_ylabel("gyroY(rad/s)")
        ax10.set_xlabel("Time(s)")
        
        ax11.set_ylabel("gyroZ(rad/s)")
        ax11.set_xlabel("Time(s)")

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
            index_bio   = np.argmax(array_Bio[:, 0]) if USEFLAG_BIO else None
            index_imu   = np.argmax(array_IMU[:, 0]) if USEFLAG_IMU else None
            index_imu_2 = np.argmax(array_IMU_2[:, 0]) if USEFLAG_IMU_2 else None
            index_wit   = np.argmax(array_Wit[:, 0]) if USEFLAG_WIT else None
            index_gps   = np.argmax(array_GPS[:, 0]) if USEFLAG_GPS else None
            # print("index_gps",index_gps)

            if index_bio and USEFLAG_BIO:
                ecg_time = array_Bio[max(0, index_bio-3000):index_bio+1, 1] /1000 # * CON.TSAMPLING
                ecg_val  = array_Bio[max(0, index_bio-3000):index_bio+1, 2]
                # ecg_time = array_Bio[0:index_bio+1, 1] * CON.TSAMPLING
                # ecg_val  = array_Bio[0:index_bio+1, 2]

                line1.set_xdata(ecg_time)
                line1.set_ydata(ecg_val)
                
                ax1.relim()
                ax1.autoscale_view()


                
                # --start-- add mtsuzuki 2024.08.11 メイン処理
                if (index_bio > CON.UPDATE_NOR_FILT_FINDPEAK) and (index_bio // CON.UPDATE_NOR_FILT_FINDPEAK > tmpUpdateCounter):
                    ecg_time_ana = array_Bio[0:index_bio+1, 1] * CON.TSAMPLING
                    ecg_val_ana  = array_Bio[0:index_bio+1, 2]

                    # --------- normalize
                    ecg_normalized = fnc_ecg_normalize(ecg_val_ana,1000,-500)
                    
                    # --------- band pass filter
                    b, a = fnc_butter_bandpass(CON.FREQ_BPF_LO, CON.FREQ_BPF_HI, 1/CON.TSAMPLING, order=4)
                    ecg_filtered = signal.filtfilt(b, a, ecg_normalized)

                    # --------- peak search
                    # peaks_idx, _ = find_peaks(ecg_filtered,
                    #                           distance=int(CON.MIN_PEAK_DIS/CON.TSAMPLING),
                    #                           height=CON.MIN_PEAK_H)
                    peaks_idx, _ = find_peaks(ecg_filtered[max(0,index_bio-30000):index_bio+1],
                                            distance=int(CON.MIN_PEAK_DIS * CON.FREQUENCY),
                                            height=CON.MIN_PEAK_H)

                    # print('max(0,index-3000):index+1:',max(0,index-3000),index+1 )
                    # --- 2024.08.30 処理速度向上のため、ピーク検出の対象とする区間を短くする
                    print("--- counter ---", tmpUpdateCounter) if CON.DEBUGFLAG_BIO else None
                    if max(0,index_bio-30000) == 0:
                        peaks_idx = peaks_idx
                    else:
                        peaks_idx = peaks_idx + max(0,index_bio-30000)

                    # print("peaks_idx:",peaks_idx)

                    # --- peaks_idxの中にすでに保存済みのインデックス番号がarray_Peakに存在した場合に削除する
                    # --- １回前の処理で得られたインデックスと重複しないようにするため
                    # tmpMatchIdx = np.isin(peaks_idx,array_Peak)
                    # peaks_idx   = np.delete(peaks_idx, np.where(tmpMatchIdx))

                    # --- 2024.09.10 add 
                    # --- 最後の１つのピークはフィルタ処理の影響で歪んだ波形のピークを拾っている可能性が
                    #     あるため、１つ手前までを使用する
                    peaks_idx = peaks_idx[:-1]
                    # print('peaks_idx:',peaks_idx)

                    # --- peaks_idxの要素でarray_Peakに保存されたインデックス番号よりも
                    #     大きいインデックスのみを取得する
                    # --- 2024.09.10 add
                    #     array_Peakに保存された最大インデックス番号に最小ピーク間隔分のインデックス幅を足した
                    #     インデックス番号より大きいインデックスをarray_Peakに；追加する
                    tmpMaxIdx_array_Peak = max(array_Peak)
                    # print('tmpMaxIdx_array_Peak:',tmpMaxIdx_array_Peak)
                    # print("peaks_idx_b4",peaks_idx)
                    # peaks_idx = peaks_idx[peaks_idx > tmpMaxIdx_array_Peak]
                    # 1回前のループで抽出したピークインデックスと小さいズレがあった場合の対応
                    peaks_idx = peaks_idx[peaks_idx > tmpMaxIdx_array_Peak + CON.MIN_PEAK_DIS * CON.FREQUENCY ]
                    # print('peaks_idx:',peaks_idx)
                    # print("peaks_idx_af",peaks_idx)

                    # --- 共有メモリに作成した配列に追加ピークインデックスを追加する処理
                    tmpSizePeakIdx_b4 = tmpSizePeakIdx
                    tmpSizePeakIdx    = tmpSizePeakIdx + len(peaks_idx)
                    array_Peak[tmpSizePeakIdx_b4:tmpSizePeakIdx] = peaks_idx
                    # print("peaks_idx:",peaks_idx)
                    # print("tmpSizePeakIdx_b4:",tmpSizePeakIdx_b4)
                    # print("tmpSizePeakIdx:",tmpSizePeakIdx)
                    # print("array_Peak:",array_Peak)

                    # --- 共有メモリに作成した配列のゼロ埋めされている部分を除外して新規の配列に代入
                    zero_idx_min = min(np.where(array_Peak[1:] == 0)[0]) + 1
                    # print("zero_idx_min:",zero_idx_min)
                    array_Peak_excludeZero = array_Peak[0:zero_idx_min]
                    # print("array_Peak_excludeZero:",array_Peak_excludeZero)
                    # np.savetxt("zzz3.txt", array_Peak_excludeZero,delimiter=",", header="array_Peak_excludeZero",comments='')


                    # print("array_Peak_excludeZero:",array_Peak_excludeZero)
                    # np.savetxt("zzz.csv", array_Peak_excludeZero,delimiter=",", header="peak_idx",comments='')

                    # --- 隣り合うピークインデックスが閾値以下の場合ピークレベルの低いほうを削除する
                    # --- 2024.08.30 ◆◆◆一旦保留◆◆◆
                    # if len(array_Peak_excludeZero) > 1:
                    #     tmpDiffIdx = np.diff(array_Peak_excludeZero) / FREQUENCY
                    #     tmpIdxToCheck = np.where(tmpDiffIdx < MIN_PEAK_DIS)
                    #     print("tmpIdxToCheck:",tmpIdxToCheck[0])
                    #     tmpIdxToDelete = []
                    #     print("len(tmpIdxToCheck):",len(tmpIdxToCheck[0]))
                    #     for ii in range(len(tmpIdxToCheck[0])):
                    #         if ecg_val[array_Peak_excludeZero[tmpIdxToCheck[ii]]] < ecg_val[array_Peak_excludeZero[tmpIdxToCheck[ii]+1]]: 
                    #             tmpIdxToDelete.append(ii)
                    #         else:
                    #             tmpIdxToDelete.append(ii+1)

                    #     print("tmpIdxToDelete:",tmpIdxToDelete)
                    #     array_Peak_excludeZero = np.delete(array_Peak_excludeZero,tmpIdxToDelete)

                    # print("array_Peak_excludeZero:",array_Peak_excludeZero)

                    # --------- R-R interval
                    # RRI_time  = ecg_time[peaks_idx[0:-1]]
                    # RRI_value = np.diff(ecg_time[peaks_idx])
                    RRI_time  = ecg_time_ana[array_Peak_excludeZero[0:-1]]
                    RRI_value = np.diff(ecg_time_ana[array_Peak_excludeZero])

                    # print("RRI_time:",RRI_time)
                    # print("RRI_value:",RRI_value)
                    # np.savetxt("zzz1.txt", RRI_time,delimiter=",", header="RRI_time",comments='')
                    # np.savetxt("zzz2.txt", RRI_value,delimiter=",", header="RRI_value",comments='')
                    
                    HR = 60./RRI_value
                    
                    # --------- remove outlier
                    tmp_RRI_value,tmp_RRI_time = fnc_remove_outlier_next_to_each_other(RRI_value , RRI_time , CON.MAX_CHANGE_RATE_RRI)
                    
                    # Clear previous plots
                    ax2.cla()
                    ax3.cla()
                    
                    # Redraw the plots
                    line3, = ax2.plot(ecg_time_ana, ecg_filtered, 'k-')
                    line3, = ax3.plot(RRI_time, HR, 'k-')
                                
                    ax2.set_ylabel("ECG Normalized (-)")

                    ax3.set_ylabel("Heart Rate(bpm)")

                    line2.set_xdata(ecg_time_ana)
                    line2.set_ydata(ecg_filtered)

                    line3.set_xdata(RRI_time)
                    line3.set_ydata(HR)

                    ax2.set_xlim(max(0,ecg_time_ana[-1] - CON.XLIM_AX2), ecg_time_ana[-1])
                    ax3.set_xlim(max(0,ecg_time_ana[-1] - CON.XLIM_AX3), ecg_time_ana[-1])

                    ax2.autoscale_view(scaley=True)
                    ax3.autoscale_view(scaley=True)
                    # print("HR:",HR)
                    lwr_limit = int(max(min(HR) if len(HR) else    0, 50) * 0.9)
                    upr_limit = int(min(max(HR) if len(HR) else 1000,100) * 1.1)
                    ax3.set_ylim(lwr_limit,upr_limit)
                    
                    # peak_times  = ecg_time[peaks_idx]
                    # peak_values = ecg_filtered[peaks_idx]
                    peak_times  = ecg_time_ana[array_Peak_excludeZero]
                    peak_values = ecg_filtered[array_Peak_excludeZero]
                    ax2.scatter(peak_times, peak_values, facecolors='none', edgecolors='red', marker='o', label='Peaks')
                    
                    # plt.draw()
                    # plt.pause(0.05)
                    
                    tmpUpdateCounter += 1

                if (index_bio > CON.UPDATE_LFHF_CCVTP):
                    # --------- calc LFHF,ccvTP
                    LFHF_time,LF,HF,TP,LFHF,ccvTP = fnc_Calc_LFHF_ccvTP(tmp_RRI_time,tmp_RRI_value,DataPlotFlag=CON.DataPlotFlag)

                    # Clear previous plots
                    ax4.cla()
                    ax5.cla()
                    
                    # Redraw the plots
                    line4, = ax4.plot(LFHF_time, LFHF,  'k-')
                    line5, = ax5.plot(LFHF_time, ccvTP, 'k-')
                                
                    ax4.set_ylabel("LFHF (-)")
                    # ax3.set_xlabel("Time(s)")

                    line4.set_xdata(LFHF_time)
                    line4.set_ydata(LFHF)

                    ax5.set_ylabel("ccvTP (-)")
                    ax5.set_xlabel("Time(s)")

                    line5.set_xdata(LFHF_time)
                    line5.set_ydata(ccvTP)

                    ax4.set_xlim(max(0,LFHF_time[-1] - CON.XLIM_AX4 - CON.XLIM_AX4/2), LFHF_time[-1] + CON.XLIM_AX4/2)
                    ax5.set_xlim(max(0,LFHF_time[-1] - CON.XLIM_AX5 - CON.XLIM_AX5/2), LFHF_time[-1] + CON.XLIM_AX5/2)

                    ax4.autoscale_view(scaley=True)
                    ax5.autoscale_view(scaley=True)
                        

                    # LF/HF,ccvTP保存用に共有メモリに書き込み
                    existing_shm_LFHF = shared_memory.SharedMemory(name=shared_mem_name_Bio_LFHF)
                    array_LFHF        = np.ndarray((len(LFHF_time), 6), dtype=np.float64, buffer=existing_shm_LFHF.buf)
                    array_LFHF[:, 0] = LFHF_time
                    array_LFHF[:, 1] = LF
                    array_LFHF[:, 2] = HF
                    array_LFHF[:, 3] = TP
                    array_LFHF[:, 4] = LFHF
                    array_LFHF[:, 5] = ccvTP
                    # print("LFHF_time:",LFHF_time)

                # -- end -- add mtsuzuki 2024.08.11 メイン処理                   



            if index_imu and USEFLAG_IMU:
                x_data = array_IMU[max(0, index_imu-300):index_imu+1, 1] / 50 # 50Hzサンプリング
                accX   = array_IMU[max(0, index_imu-300):index_imu+1, 3]
                accY   = array_IMU[max(0, index_imu-300):index_imu+1, 4]
                accZ   = array_IMU[max(0, index_imu-300):index_imu+1, 5]
                gyroX  = array_IMU[max(0, index_imu-300):index_imu+1, 6]
                gyroY  = array_IMU[max(0, index_imu-300):index_imu+1, 7]
                gyroZ  = array_IMU[max(0, index_imu-300):index_imu+1, 8]
                # quatR = array[max(0, index_imu-300):index_imu+1, 9]
                # quatI = array[max(0, index_imu-300):index_imu+1, 10]
                # quatJ = array[max(0, index_imu-300):index_imu+1, 11]
                # quatK = array[max(0, index_imu-300):index_imu+1, 12]
                
                line6.set_xdata(x_data)
                line6.set_ydata(accX)

                line7.set_xdata(x_data)
                line7.set_ydata(accY)
                
                line8.set_xdata(x_data)
                line8.set_ydata(accZ)

                # line9.set_xdata(x_data)
                # line9.set_ydata(gyroX)

                # line10.set_xdata(x_data)
                # line10.set_ydata(gyroY)
                
                # line11.set_xdata(x_data)
                # line11.set_ydata(gyroZ)

                # line7.set_xdata(x_data)
                # line7.set_ydata(quatR)

                # line8.set_xdata(x_data)
                # line8.set_ydata(quatI)

                # line9.set_xdata(x_data)
                # line9.set_ydata(quatJ)

                # line10.set_xdata(x_data)
                # line10.set_ydata(quatK)

                ax6.relim()
                ax6.autoscale_view()

                ax7.relim()
                ax7.autoscale_view()

                ax8.relim()
                ax8.autoscale_view()

                # ax9.relim()
                # ax9.autoscale_view()

                # ax10.relim()
                # ax10.autoscale_view()

                # ax11.relim()
                # ax11.autoscale_view()

                # ax7.relim()
                # ax7.autoscale_view()

                # ax8.relim()
                # ax8.autoscale_view()

                # ax9.relim()
                # ax9.autoscale_view()

                # ax10.relim()
                # ax10.autoscale_view()

            if index_imu_2 and USEFLAG_IMU_2:
                x_data_2 = array_IMU_2[max(0, index_imu_2-300):index_imu_2+1, 1] / 50 # 50Hzサンプリング
                accX_2   = array_IMU_2[max(0, index_imu_2-300):index_imu_2+1, 3]
                accY_2   = array_IMU_2[max(0, index_imu_2-300):index_imu_2+1, 4]
                accZ_2   = array_IMU_2[max(0, index_imu_2-300):index_imu_2+1, 5]

                line9.set_xdata(x_data_2)
                line9.set_ydata(accX_2)

                line10.set_xdata(x_data_2)
                line10.set_ydata(accY_2)
                
                line11.set_xdata(x_data_2)
                line11.set_ydata(accZ_2)

                ax9.relim()
                ax9.autoscale_view()

                ax10.relim()
                ax10.autoscale_view()

                ax11.relim()
                ax11.autoscale_view()

            if index_wit and USEFLAG_WIT:
                x_data = array_Wit[max(0, index_wit-300):index_wit+1, 1]
                accX   = array_Wit[max(0, index_wit-300):index_wit+1, 2]
                accY   = array_Wit[max(0, index_wit-300):index_wit+1, 3]
                accZ   = array_Wit[max(0, index_wit-300):index_wit+1, 4]
                gyroX  = array_Wit[max(0, index_wit-300):index_wit+1, 5]
                gyroY  = array_Wit[max(0, index_wit-300):index_wit+1, 6]
                gyroZ  = array_Wit[max(0, index_wit-300):index_wit+1, 7]
                
                line6.set_xdata(x_data)
                line6.set_ydata(accX)

                line7.set_xdata(x_data)
                line7.set_ydata(accY)
                
                line8.set_xdata(x_data)
                line8.set_ydata(accZ)

                line9.set_xdata(x_data)
                line9.set_ydata(gyroX)

                line10.set_xdata(x_data)
                line10.set_ydata(gyroY)
                
                line11.set_xdata(x_data)
                line11.set_ydata(gyroZ)

                ax6.relim()
                ax6.autoscale_view()

                ax7.relim()
                ax7.autoscale_view()

                ax8.relim()
                ax8.autoscale_view()

                ax9.relim()
                ax9.autoscale_view()

                ax10.relim()
                ax10.autoscale_view()

                ax11.relim()
                ax11.autoscale_view()

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

                    # if CON.USEFLAG_BIO:
                    if USEFLAG_BIO:
                        existing_shm_LFHF_gps = shared_memory.SharedMemory(name=shared_mem_name_Bio_LFHF)
                        buffer_size = existing_shm_LFHF_gps.size
                        array_LFHF_gps = np.ndarray((buffer_size // (6 * np.dtype(np.float64).itemsize), 6), dtype=np.float64, buffer=existing_shm_LFHF_gps.buf)
                        
                        # LFHFを初期化
                        LFHF_gps = np.zeros((array_LFHF_gps.shape[0], 2))
                        LFHF_gps[:,0] = array_LFHF_gps[:,0]
                        LFHF_gps[:,1] = array_LFHF_gps[:,4]

                        # GPSプロットにLFHFのカラーマップを表示するための準備
                        dt_gps = round(array_gps_ExcludeZero_interp[1,2] - array_gps_ExcludeZero_interp[0,2],8) # 小数点以下8桁に四捨五入
                        # LF/HF算出のウインドウ幅のどの時刻のデータととらえるかの違いでGPS情報の初期のインデックス番号を変更
                        if CON.DataPlotFlag == 'first':
                            ini_index_forPlot_gps_with_ECG = 0
                        elif CON.DataPlotFlag == 'mid':
                            ini_index_forPlot_gps_with_ECG = int(CON.WINDOW_DURATION / 2 / dt_gps)
                        elif CON.DataPlotFlag == 'last':
                            ini_index_forPlot_gps_with_ECG = int(CON.WINDOW_DURATION    / dt_gps)

                        # start:end:step の順 最初の50秒はスキップするためstart部分にstep分を記載 以降は５秒(=indexは50)ごと
                        array_gps_downsampled = array_gps_ExcludeZero_interp[ini_index_forPlot_gps_with_ECG::int(CON.STEP_SIZE/dt_gps)]                

                        time_lon_lat_downsampled = np.column_stack((array_gps_downsampled[:, 2],array_gps_downsampled[:, 4],array_gps_downsampled[:, 3]))

                        if index_gps > CON.WINDOW_DURATION/0.1 + tmpUpdateCounter_LFHF_GPS * int(CON.STEP_SIZE/dt_gps):
                            # print(index,":into if")

                            LFHF_gps_index = min(np.where(LFHF_gps[1:,0] == 0)[0]) # 余分な行を削除
                            LFHF_gps       = LFHF_gps[:LFHF_gps_index + 1]

                            # LFHFと緯度経度のデータを結合
                            # LFHFのほうが行数は少ないはずだが、両者の行数の小さいほうに合わせるように結合する
                            min_rows = min(LFHF_gps.shape[0], time_lon_lat_downsampled.shape[0])
                            LFHF_trimmed   = LFHF_gps[:min_rows]
                            lonlat_trimmed = time_lon_lat_downsampled[:min_rows]

                            # LFHF_trimmedの時間軸を取得
                            time_LFHF = LFHF_trimmed[:, 0]
                            # lonlat_trimmedの時間軸を取得
                            time_lonlat = lonlat_trimmed[:, 0]

                            # 緯度と経度の補間関数を作成
                            interp_lon = interp1d(time_lonlat, lonlat_trimmed[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
                            interp_lat = interp1d(time_lonlat, lonlat_trimmed[:, 2], kind='linear', bounds_error=False, fill_value="extrapolate")

                            # LFHF_trimmedの時間軸に合わせて補間
                            lon_interpolated = interp_lon(time_LFHF)
                            lat_interpolated = interp_lat(time_LFHF)

                            # 補間した緯度経度データを2次元配列に変換
                            lon_interpolated = lon_interpolated[:, np.newaxis]
                            lat_interpolated = lat_interpolated[:, np.newaxis]

                            # LFHF_trimmedと補間した緯度経度データを結合
                            time_LFHF_lon_lat = np.concatenate((LFHF_trimmed, lon_interpolated,lat_interpolated), axis=1)
                            print("time LFHF lon lat:",time_LFHF_lon_lat[:,1])

                            normlized_color = time_LFHF_lon_lat[:,1]

                            # LineCollectionを使用して色を設定
                            points = np.array([time_LFHF_lon_lat[:,2], time_LFHF_lon_lat[:,3]]).T.reshape(-1, 1, 2)
                            segments = np.concatenate([points[:-1], points[1:]], axis=1)
                            lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
                            lc.set_array(time_LFHF_lon_lat[:,1])
                            ax12.add_collection(lc)
                            
                            # カラーバーを追加
                            if not colorbar_added:
                                colorbar = plt.colorbar(lc, ax=ax12)
                                colorbar_added = True
                            else:
                                colorbar.mappable.set_clim(vmin=0, vmax=20)
                                
                            tmpUpdateCounter_LFHF_GPS += 1

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