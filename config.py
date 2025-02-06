# config.py

import numpy as np

"""使用デバイスフラグ"""
USEFLAG_BIO   = 0
USEFLAG_IMU   = 0
USEFLAG_IMU_2 = 0
USEFLAG_WIT   = 0
USEFLAG_GPS   = 0
USEFLAG_FTF   = 0
USEFLAG_CAN   = 0


"""グラフ表示フラグ"""
# レース車両への搭載時など、CAN情報を出力し、別システムでデータモニタリングする場合は
# グラフ表示が不要になる
FIGFLAG  = 1 # 0：表示しない 1：表示する 

"""Biosignal用の共有メモリ設定"""
MAX_HOURS   = 2
FREQUENCY   = 1000 
BIO_SHAPE   = (int(MAX_HOURS * 60 * 60 * FREQUENCY), 3)  # 3列の形状に変更
BIO_DTYPE   = np.float64
FILENAME    = "biosignal_data.csv"

"""Biosignal用Bluetooth設定"""
PLUX_PATH = ".\\PLUX-API-Python3\\Win64_310"

# BT_ADDRESS  = "88:6B:0F:D9:18:80" # bitalino
BT_ADDRESS  = "00:07:80:4D:2D:71" # R-Frontier 1号機
# BT_ADDRESS  = "00:07:80:4D:2D:39" # R-Frontier 3号機


DURATION = int(MAX_HOURS*60*60)
# CODE     = 0x01
CODE     = 0x02

"""IMU用共有メモリ設定"""
# Shared memory configuration
IMU_SHAPE = (int(MAX_HOURS * 60 * 60 * 100), 14)  # 14列: time_diff, index, elapsed_time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, qR, qI, qJ, qK, voltage
IMU_DTYPE = np.float32

"""IMU用Bluetooth設定"""
BLE_DEVICE_NAME = "ESP32_BNO085"

# BLE_DEVICE_ADDR = "64:E8:33:87:12:12" # For Windows
BLE_DEVICE_ADDR = "F0:F5:BD:19:0B:B6" #IMU1 For Windows
# BLE_DEVICE_ADDR = "54:32:04:22:64:3E" #IMU2 For Windows
# BLE_DEVICE_ADDR = "8C:BF:EA:CD:7E:86" #IMU3 For Windows
# BLE_DEVICE_ADDR = "8C:BF:EA:CB:A9:9A" #IMU4 For Windows
# BLE_DEVICE_ADDR = "F0:F5:BD:19:0B:B6" #IMU5 For Windows
# BLE_DEVICE_ADDR = "54:32:04:33:2D:D2" #IMU6 For Windows
# BLE_DEVICE_ADDR = "8C:BF:EA:CB:D0:2A" #IMU7 For Windows
# BLE_DEVICE_ADDR = "8C:BF:EA:CD:78:E2" #IMU8 For Windows
# BLE_DEVICE_ADDR = "DFF6D319-E61A-00BE-FE46-0258EDFCA422" # IMU① For Mac　
# BLE_DEVICE_ADDR = "897015BD-EC6C-2746-2942-C5DB0C5D36DD" # IMU② For Mac

# BLE_DEVICE_ADDR_2 = "64:E8:33:87:12:12" # For Windows
# BLE_DEVICE_ADDR_2 = "F0:F5:BD:19:0B:B6" #IMU1 For Windows
BLE_DEVICE_ADDR_2 = "54:32:04:22:64:3E" #IMU2 For Windows
# BLE_DEVICE_ADDR_2 = "8C:BF:EA:CD:7E:86" #IMU3 For Windows
# BLE_DEVICE_ADDR_2 = "8C:BF:EA:CB:A9:9A" #IMU4 For Windows
# BLE_DEVICE_ADDR_2 = "F0:F5:BD:19:0B:B6" #IMU5 For Windows
# BLE_DEVICE_ADDR_2 = "54:32:04:33:2D:D2" #IMU6 For Windows
# BLE_DEVICE_ADDR_2 = "8C:BF:EA:CB:D0:2A" #IMU7 For Windows
# BLE_DEVICE_ADDR_2 = "8C:BF:EA:CD:78:E2" #IMU8 For Windows
# BLE_DEVICE_ADDR_2 = "DFF6D319-E61A-00BE-FE46-0258EDFCA422" # IMU① For Mac　
# BLE_DEVICE_ADDR_2 = "897015BD-EC6C-2746-2942-C5DB0C5D36DD" # IMU② For Mac

SERVICE_UUID    = "180D"
ACCEL_CHARACTERISTIC_UUID = "2A37"

"""WitMotion用共有メモリ設定"""
# Shared memory configuration
WIT_SHAPE = (int(MAX_HOURS * 60 * 60 * 100), 11)  # 11列:time_diff, index, ax, ay, az, wx, wy, wz, roll, pitch, yaw
WIT_DTYPE = np.float32

"""WitMotion用Bluetooth設定"""
WIT_BLE_DEVICE_NAME             = "WT901BLE68"
WIT_BLE_DEVICE_ADDR             = "cb:4d:4f:d7:80:41" #"DA002BDE-59FC-8646-85D1-7D02114635F3"
WIT_ACCEL_CHARACTERISTIC_UUID   = "0000ffe4-0000-1000-8000-00805f9a34fb"
WIT_WRITE_UUID                  = "0000ffe9-0000-1000-8000-00805f9a34fb"

"""GPS用共有メモリ設定"""
# GPSはあらかじめPCとペアリングしてデバイス（COM＊）として認識できるようにしておくこと。
# Shared memory configuration
GPS_SHAPE = (int(MAX_HOURS * 60 * 60 * 10), 5)  #5列: pc_time, utc_date, utc_time, lat, lon 
GPS_DTYPE = np.float64

"""GPS用Bluetooth設定"""
# GPS_PORT        = "COM9" #README_hmを参照して設定
# GPS_PORT        = "COM5" # mtsuzki home PC & stick PC
# GPS_PORT        = "COM6" # Tough Book Takakura
GPS_PORT        = "COM14" # Tough Book GPS_1 
# GPS_PORT        = "COM16" # Tough Book GPS_2
GPS_BAUDRATE    = 9600
GPS_TIMEOUT     = 1

"""ecgデータ処理用パラメータ"""
# パラメータ 基本情報
TSAMPLING = 1/FREQUENCY
COLECG    = 2
COLACCX   = 3
COLACCY   = 4
COLACCZ   = 5

# バンドパスフィルタ
FREQ_BPF_LO =  1 # [Hz]
FREQ_BPF_HI = 40 # [Hz]

# パラメータ ピーク検出
MAX_PEAK_WID = 2.0
MIN_PEAK_DIS = 0.4
MIN_PEAK_PRO = 400
MIN_PEAK_H   = 500

# パラメータ 外れ値処理
MAX_CHANGE_RATE_RRI = 20 # [%]

# パラメータ 処理実行タイミング
UPDATE_NOR_FILT_FINDPEAK =   1 * FREQUENCY # ECG正規化,フィルタリング,ピーク検出
UPDATE_LFHF_CCVTP        = 100 * FREQUENCY # LFHF,ccvTP

# パラメータ グラフ表示
XLIM_AX2 = 10  # sec グラフ表示範囲更新用 ECG-Normalized w/R-Peak
XLIM_AX3 = 10  # sec グラフ表示範囲更新用 Heart Rate
XLIM_AX4 = 100 # sec グラフ表示範囲更新用 LFHF
XLIM_AX5 = 100 # sec グラフ表示範囲更新用 ccvTP

# パラメータ LF,HF,TP計算用
WINDOW_DURATION = 100    # 100秒で固定
STEP_SIZE       = 5      # 秒
REFFREQ         = 1000.0 # どんな信号が入力されても1000Hzでリサンプリング
BIO_SHAPE_LFHF  = (int(np.fix((int(MAX_HOURS * 60 * 60) - WINDOW_DURATION) / STEP_SIZE)), 6)  # 6列 TIme,LF,HF,TP,LF/HF,ccvTP
BIO_SHAPE_PEAK  = (int(MAX_HOURS * 60 * 60 * FREQUENCY))

# GPS + LF/HF プロット用パラメータ
DataPlotFlag = 'last' # 'first' , 'mid' , 'last' から選択 LF/HFを算出するウインドウ幅のどの時刻のデータとみなすかのフラグ

"""CANFD用パラメータ"""
"""
共有メモリごとにCANID,データサイズを設定
[
 bio_shared_mem.name,
 bio_shared_mem_LFHF.name,
 bio_shared_mem_Peak.name,
 imu_shared_mem.name,
 wit_shared_mem.name,
 gps_shared_mem.name
]
"""

# CAN_IDS = [ [1024, 1025, 1026], 
#             [1040, 1041, 1042, 1043, 1044, 1045],
#             [1046, 1047, 1048],
#         imu [1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1072, 1073, 1074],
#         wit [1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1104], 
#             [1120, 1121, 1122, 1123, 1124]

# CAN_IDS = [
#             [1024, 1025, 1026],
#             [1040, 1041, 1042, 1043, 1044, 1045],
#             [1046, 1047, 1048],
#        imu  [1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069],
#        wit  [1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098],
#             [1120, 1121, 1122, 1123, 1124]
 


# CAN_IDS   = [0x100, 0x101, 0x102, 0x103, 0x104, 0x105] 
CAN_IDS   = [
             [0x400, 0x401, 0x402],
             [0x410, 0x411, 0x412, 0x413, 0x414, 0x415],
             [0x416, 0x417, 0x418],
             [0x420, 0x421, 0x422, 0x423, 0x424, 0x425, 0x426, 0x427, 0x428, 0x429, 0x42A, 0x42B, 0x42C, 0x42D],
             [0x440, 0x441, 0x442, 0x443, 0x444, 0x445, 0x446, 0x447, 0x448, 0x449, 0x44A],
             [0x460, 0x461, 0x462, 0x463, 0x464]
            ] 
DATA_NUMS = [3*8,   6*8,   3*8,   13*4,  11*4,   5*8]

RECEIVE_FLAG = 0 #0:can0で送信のみ、1:can0で送信,can1で受信
# BAUD_RATES = [(1000000, 5000000)] #アービトレーション,データ送信)
BAUD_RATES = [(5000000, 5000000)] #アービトレーション,データ送信)

"""酔い低減アイテム用パラメータ"""
FTF_SHAPE = (int(MAX_HOURS * 60 * 60 * 100), 2)  #2列: pc_time, data 
FTF_DTYPE = np.float32

M5_CONTROLLER_PORT      = "COM10"
M5_CONTROLLER_BAUDRATE  = 115200 
M5_CONTROLLER_TIMEOUT   = 0.01

FTF_PORT        = "COM8"
FTF_BAUDRATE    = 921600
FTF_TIMEOUT     = 0.1

M5_COUNTER_PORT     = "COM11"
M5_COUNTER_BAUDRATE = 115200 
M5_COUNTER_TIMEOUT  = 0.01

DEBUGFLAG_BIO      = 0
DEBUGFLAG_CAN_SEND = 0
DEBUGFLAG_CAN_RCVE = 0
DEBUGFLAG_CAN_RCVE_VERIFICATION = 0

SAVEINTERVAL = 30 # ファイル保存間隔 [sec]