# main.py

# 最初に仮想環境をアクティベートすること ##########################
# 
# コマンドプロンプト例
#   C:\Users\jan_d\.py310\Scripts\activate	 コマンドプロンプトの場合
#   C:\Users\jan_d\.py310\Scripts\Activate.ps1 PowerShellの場合
#
# RaspberryPi4 -> xhost +local:

import sys
from multiprocessing import Process, shared_memory
from datetime import datetime
import numpy as np
import time
from config import *

from gps_bt_receiver import start_gps_receiver
from shared_memory_manager import create_shared_memory
from plotter_gps import start_plotting_one_figure
# add mtsuzuki
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))

import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from config_manager import read_config, write_config

from multiprocessing import Lock

# 排他処理用設定
global_lock = Lock()

# グローバル変数として保存ディレクトリを保持
save_directory = None

# グローバル変数としてプロセスを保持
gps_process = None
plot_gps_process = None
plot_one_figure = None

config = read_config()
USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])
FIGFLAG	           = int(config['DEFAULT']['FIGFLAG'])

save_thread_running = True

# 前回保存したインデックスを記録
last_saved_indices = {
    'gps': 0,
}

def stop_processes():

	if USEFLAG_GPS:
		gps_process.terminate()
		# if FIGFLAG:
		# 	plot_gps_process.terminate()

	plot_one_figure_process.terminate()

	print("Acquisition stopped by user.")



def main():
	global gps_process, plot_gps_process
	global gps_array
	global plot_one_figure_process


	config = read_config()  # ここで最新の設定を読み込む
	# USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])
	# FIGFLAG            = int(config['DEFAULT']['FIGFLAG'])
	USEFLAG_GPS        = 1
	FIGFLAG            = 1

	start_time = int(time.time() * 1000)

    # 定期的にデータを保存するスレッドを開始
	save_thread = threading.Thread(target=save_data_periodically, args=(SAVEINTERVAL,))
	save_thread.start()

	"""共有メモリの作成"""
	gps_shared_mem, gps_array = create_shared_memory(GPS_SHAPE, GPS_DTYPE)
	gps_shared_mem_name = gps_shared_mem.name
	flg_shared_mem, flg_array = create_shared_memory(FLG_SHAPE, FLG_DTYPE)
	flg_shared_mem_name = gps_shared_mem.name



	"""プロセスの定義"""
	if USEFLAG_GPS:
		gps_process				 = Process(target=start_gps_receiver, 			args=(gps_shared_mem_name,start_time))
		# plot_gps_process		 = Process(target=start_plotting_gps, 			args=(gps_shared_mem_name,bio_shared_mem_LFHF_name,))

	plot_one_figure_process = Process(target=start_plotting_one_figure,		    args=(gps_shared_mem_name,))

	"""実行"""
	try:
		"""プロセススタート"""
		if USEFLAG_GPS:
			gps_process.start()
			# if FIGFLAG:
			# 	plot_gps_process.start()
		if FIGFLAG:
			print("Plot Process started")
			plot_one_figure_process.start()

		if USEFLAG_GPS:
			gps_process.join()
			# if FIGFLAG:
			# 	plot_gps_process.terminate()

		plot_one_figure_process.join()


	except KeyboardInterrupt:
		"""キーボードが押されたら終了する"""
		stop_processes()

	finally:
		"""終了前に保存処理する"""
		print("Saving Data...")
		try:
			save_data()  # 終了時にもデータを保存

		except Exception as e:
			print(f"An error occurred while saving data: {e}")

		"""終了前に共有メモリを解放する"""
		try:
			if USEFLAG_GPS:
				gps_shared_mem.close()
				gps_shared_mem.unlink()
				print(f"Shared memories(GPS	  ) '{gps_shared_mem_name}' have been successfully deleted.")

		except FileNotFoundError:
			if USEFLAG_GPS:
				print(f"Shared memories(GPS) '{gps_shared_mem_name}' do not exist or have already been deleted.")

		except Exception as e:
			print(f"An error occurred: {e}")

		print("Finish!!!")
		print("Wait... Saving Data")



# 定期ファイル保存機能追加
def save_data_periodically(interval=60):
    # while True:
    #     time.sleep(interval)
    #     save_data()
	global save_thread_running
	while save_thread_running:
		time.sleep(interval)
		save_data()

def save_data():
	global save_directory

	config = read_config()
	USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])


	with global_lock:
		print("Saving Data (periodically)...")
		now = datetime.now()
		savedatetime = now.strftime("%Y%m%d%H%M%S")

		if save_directory is None:
			save_directory = u'./savedata/SaveData_' + savedatetime
			os.makedirs(save_directory, exist_ok=True)

		savedir = save_directory
		fmt = '%.18e'  # 科学記数法で出力するフォーマット

		try:


			if USEFLAG_GPS:
				# gps_index = int(np.max(gps_array[:, 1]))
				gps_index = int(np.argmax(gps_array[:, 0]))
				gps_data_to_save = gps_array[last_saved_indices['gps']:gps_index + 1]
				file_path = savedir + u"/gps_data.txt"
				if not os.path.exists(file_path):
					header = "pc_time, utc_date, utc_time, lat, lon"
				else:
					header = ''
				with open(savedir + u"/gps_data.txt", 'a') as f:
					np.savetxt(f, gps_data_to_save, delimiter=",", header=header, comments='', fmt=fmt)
				print("Data appended to 'gps_data.txt'.")
				last_saved_indices['gps'] = gps_index + 1
                
                # htmlファイルに地図データを保存
				lon = gps_data_to_save[:, 4]  # longitude
				lat = gps_data_to_save[:, 3]  # latitude
				# update_map(lat, lon, map_file=savedir + u"/gps_data.html")

		except Exception as e:
			print(f"An error occurred while saving data: {e}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)