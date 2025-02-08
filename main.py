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

def main():
	global gps_process, plot_gps_process
	global gps_array
	global plot_one_figure_process


	config = read_config()  # ここで最新の設定を読み込む
	USEFLAG_GPS        = int(config['DEFAULT']['USEFLAG_GPS'])
	FIGFLAG            = int(config['DEFAULT']['FIGFLAG'])

	start_time = int(time.time() * 1000)

    # 定期的にデータを保存するスレッドを開始
	save_thread = threading.Thread(target=save_data_periodically, args=(SAVEINTERVAL,))
	save_thread.start()

	"""共有メモリの作成"""
	#if USEFLAG_GPS:
	gps_shared_mem, gps_array = create_shared_memory(GPS_SHAPE, GPS_DTYPE)
	gps_shared_mem_name = gps_shared_mem.name

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

def stop_processes():

	if USEFLAG_GPS:
		gps_process.terminate()
		# if FIGFLAG:
		# 	plot_gps_process.terminate()

	plot_one_figure_process.terminate()

	print("Acquisition stopped by user.")

def run_main():
	main_thread = threading.Thread(target=main)
	main_thread.start()
	return main_thread

def stop_main(main_thread):
	global save_thread_running
	if main_thread.is_alive():

		if USEFLAG_GPS:
			gps_process.terminate()
			# if FIGFLAG:
			# 	plot_gps_process.terminate()

		plot_one_figure_process.terminate()

		# save_thread.terminate()
		save_thread_running = False
		print("Save thread stopped.")

		print("Acquisition stopped by user.")

class MainApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Main Function Controller")

		# 画面の大きさを取得
		screen_width = root.winfo_screenwidth()
		screen_height = root.winfo_screenheight()

		# ウインドウの大きさと位置を設定
		window_width = screen_width // 2
		window_height = screen_height // 10 * 1
		window_pos_x = 0
		window_pos_y = 0
		self.root.geometry(f"{window_width}x{window_height}+{window_pos_x}+{window_pos_y}")
		
		self.main_thread = None

		# 開始・停止マークの画像を読み込む
		self.Start_icon = Image.open("./image/GUI/Start.png")
		self.Start_icon = self.Start_icon.resize((int(window_height/5), int(window_height/5)), Image.Resampling.LANCZOS)  # 画像のサイズを変更
		self.Start_icon = ImageTk.PhotoImage(self.Start_icon)
		self.Stop_icon  = Image.open("./image/GUI/Stop.png")
		self.Stop_icon  = self.Stop_icon.resize((int(window_height/5), int(window_height/5)), Image.Resampling.LANCZOS)  # 画像のサイズを変更
		self.Stop_icon  = ImageTk.PhotoImage(self.Stop_icon)

		self.start_button = tk.Button(root, text="  Start Measurement", command=self.start_main,
									  image=self.Start_icon, compound="left", width=200, height=window_height/5)
		self.start_button.grid(row=0, column=0, padx=10, pady=10)
		
		self.stop_button = tk.Button(root, text="  Stop Measurement", command=self.stop_main,
									 image=self.Stop_icon, compound="left", width=200, height=window_height/5)
		self.stop_button.grid(row=1, column=0, padx=10, pady=10)
		
		# チェックボックスを追加
		self.config = read_config()
  
		self.useflag_bio_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_BIO']))
		self.useflag_bio_checkbox = tk.Checkbutton(root, text="USEFLAG_BIO", variable=self.useflag_bio_var, command=self.update_config)
		self.useflag_bio_checkbox.grid(row=0, column=1, padx=10, pady=10, sticky="w")

		self.useflag_imu_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_IMU']))
		self.useflag_imu_checkbox = tk.Checkbutton(root, text="USEFLAG_IMU", variable=self.useflag_imu_var, command=self.update_config)
		self.useflag_imu_checkbox.grid(row=0, column=2, padx=10, pady=10, sticky="w")
 
		self.useflag_imu_var_2 = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_IMU_2']))
		self.useflag_imu_checkbox_2 = tk.Checkbutton(root, text="USEFLAG_IMU_2", variable=self.useflag_imu_var_2, command=self.update_config)
		self.useflag_imu_checkbox_2.grid(row=0, column=3, padx=10, pady=10, sticky="w")

		self.useflag_wit_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_WIT']))
		self.useflag_wit_checkbox = tk.Checkbutton(root, text="USEFLAG_WIT", variable=self.useflag_wit_var, command=self.update_config)
		self.useflag_wit_checkbox.grid(row=0, column=4, padx=10, pady=10, sticky="w")

		self.useflag_gps_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_GPS']))
		self.useflag_gps_checkbox = tk.Checkbutton(root, text="USEFLAG_GPS", variable=self.useflag_gps_var, command=self.update_config)
		self.useflag_gps_checkbox.grid(row=0, column=5, padx=10, pady=10, sticky="w")

		self.useflag_ftf_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_FTF']))
		self.useflag_ftf_checkbox = tk.Checkbutton(root, text="USEFLAG_FTF", variable=self.useflag_ftf_var, command=self.update_config)
		self.useflag_ftf_checkbox.grid(row=1, column=1, padx=10, pady=10, sticky="w")

		self.useflag_can_var = tk.IntVar(value=int(self.config['DEFAULT']['USEFLAG_CAN']))
		self.useflag_can_checkbox = tk.Checkbutton(root, text="USEFLAG_CAN", variable=self.useflag_can_var, command=self.update_config)
		self.useflag_can_checkbox.grid(row=1, column=2, padx=10, pady=10, sticky="w")

		self.useflag_fig_var = tk.IntVar(value=int(self.config['DEFAULT']['FIGFLAG']))
		self.useflag_fig_checkbox = tk.Checkbutton(root, text="FIGFLAG", variable=self.useflag_fig_var, command=self.update_config)
		self.useflag_fig_checkbox.grid(row=1, column=3, padx=10, pady=10, sticky="w")

	def update_config(self):
		self.config['DEFAULT']['USEFLAG_BIO']        = str(self.useflag_bio_var.get())
		self.config['DEFAULT']['USEFLAG_IMU']        = str(self.useflag_imu_var.get())
		self.config['DEFAULT']['USEFLAG_IMU_2']      = str(self.useflag_imu_var_2.get())
		self.config['DEFAULT']['USEFLAG_WIT']        = str(self.useflag_wit_var.get())
		self.config['DEFAULT']['USEFLAG_GPS']        = str(self.useflag_gps_var.get())
		self.config['DEFAULT']['USEFLAG_FTF']        = str(self.useflag_ftf_var.get())
		self.config['DEFAULT']['USEFLAG_CAN']        = str(self.useflag_can_var.get())
		self.config['DEFAULT']['FIGFLAG']	         = str(self.useflag_fig_var.get())
		write_config(self.config)

	def start_main(self):
		if self.main_thread is None or not self.main_thread.is_alive():
			self.main_thread = run_main()
			messagebox.showinfo("Info", "Main function started.")
		else:
			messagebox.showwarning("Warning", "Main function is already running.")



	def stop_main(self):
		if self.main_thread is not None and self.main_thread.is_alive():
			stop_main(self.main_thread)
			messagebox.showinfo("Info", "Main function stopped.")
		else:
			messagebox.showwarning("Warning", "Main function is not running.")


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
	root = tk.Tk()
	app = MainApp(root)
	root.mainloop()