import serial
import re
import asyncio
import numpy as np
from multiprocessing import shared_memory
from datetime import datetime
import config as CON
import time

# バッファの長さを指定する（秒単位）
BUFFER_DURATION = CON.DURATION  # 例：600秒 = 10分

# データ受信レート（Hz単位）
RECEIVE_RATE = 10  # 例：10Hzでデータを受信

# バッファサイズを計算
BUFFER_SIZE = BUFFER_DURATION * RECEIVE_RATE

# GPSからのNMEAデータを解析する関数
def parse_gprmc(sentence):
    match = re.match(
        r'^\$GNRMC,(\d{6}\.\d+),([AV]),(\d{2})(\d{2}\.\d+),([NS]),(\d{3})(\d{2}\.\d+),([EW]),.*?,(\d{6}),', sentence)
    if match:
        time_utc = match.group(1)  # UTC時刻
        status = match.group(2)
        lat_deg = match.group(3)
        lat_min = match.group(4)
        lat_dir = match.group(5)
        lon_deg = match.group(6)
        lon_min = match.group(7)
        lon_dir = match.group(8)
        date_utc = match.group(9)  # UTC日付

        if status == 'A':  # データが有効な場合
            latitude = (float(lat_deg) + float(lat_min) / 60.0) * (-1 if lat_dir == 'S' else 1)
            longitude = (float(lon_deg) + float(lon_min) / 60.0) * (-1 if lon_dir == 'W' else 1)
            
            # UTC時刻を秒単位に変換
            hours = int(time_utc[:2])
            minutes = int(time_utc[2:4])
            seconds = float(time_utc[4:])
            utc_time_seconds = np.float64(hours * 3600 + minutes * 60 + seconds)

            # UTC日付を"yyyymmdd"として整数化（64ビットに収まる範囲で）
            day = int(date_utc[:2])
            month = int(date_utc[2:4])
            year = int('20' + date_utc[4:6])  # 20xx年と仮定
            utc_date = np.float64(year * 10000 + month * 100 + day)

            return utc_date, utc_time_seconds, latitude, longitude
    return None, None, None, None

async def read_gps_data(serial_port, array, start_time):
    try:
        while True:
            # データを非同期に読み取る
            line = await asyncio.to_thread(serial_port.readline)
            line = line.decode('ascii', errors='replace').strip()

            if line.startswith('$GNRMC'):
                utc_date, utc_time_seconds, lat, lon = parse_gprmc(line)
                if utc_date is not None and utc_time_seconds is not None and lat is not None and lon is not None:

                    # current_time = datetime.now()
                    # pc_time = (current_time - start_time).total_seconds()
                    current_time = int(time.time() * 1000)
                    elapsed_time = np.float32(current_time - start_time)  # float32で記録
                    # 共有メモリに保存
                    index = np.where(array[:, 0] == 0)[0]
                    if index.size > 0:
                        index = index[0]
                        array[index, :] = [elapsed_time, utc_date, utc_time_seconds, lat, lon]

                    if index >= BUFFER_SIZE:
                        print("Buffer full. Stopping logging.")
                        return

    except asyncio.CancelledError:
        # 非同期処理がキャンセルされた場合
        print("Logging stopped by user.")

async def log_gps_data(shared_mem_name, start_time):
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    array = np.ndarray(CON.GPS_SHAPE, dtype=CON.GPS_DTYPE, buffer=existing_shm.buf)
    with serial.Serial(CON.GPS_PORT, CON.GPS_BAUDRATE, timeout=CON.GPS_TIMEOUT) as ser:
        await read_gps_data(ser, array, start_time)

def start_gps_receiver(shared_mem_name, start_time):
    try:
        asyncio.run(log_gps_data(shared_mem_name, start_time))
    except KeyboardInterrupt:
        print("GPS Process: Acquisition stopped by user.")
