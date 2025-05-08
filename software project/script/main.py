import tkinter as tk
from tkinter import ttk, messagebox
import folium
import webbrowser
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import os
from geopy.geocoders import Nominatim
import requests


class AutonomousDrivingSim:
    def __init__(self):
        self.map_root = tk.Tk()
        self.video_root = tk.Toplevel(self.map_root)
        self.map_root.title("地图与导航")
        self.video_root.title("实时障碍物检测")
        self.map_root.geometry("800x600")
        self.video_root.geometry("600x400")

        self.model = YOLO("yolov8n.pt")
        self.running = False
        self.geolocator = Nominatim(user_agent="autonomous_sim")
        self.amap_api_key = "0630c202107253fbbfe476cdce301735"  # 替换为你的高德API密钥0630c202107253fbbfe476cdce301735
        self.current_frame = None

        self.setup_map_gui()
        self.setup_video_gui()
        self.cap = cv2.VideoCapture(0)

        self.map_root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_map_gui(self):
        self.search_frame = ttk.Frame(self.map_root)
        self.search_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(self.search_frame, text="起点：").pack(side="left")
        self.start_entry = ttk.Entry(self.search_frame)
        self.start_entry.pack(side="left", padx=5)

        ttk.Label(self.search_frame, text="终点：").pack(side="left")
        self.end_entry = ttk.Entry(self.search_frame)
        self.end_entry.pack(side="left", padx=5)

        ttk.Label(self.search_frame, text="模式：").pack(side="left")
        self.mode_var = tk.StringVar(value="driving")
        ttk.Radiobutton(self.search_frame, text="汽车", value="driving", variable=self.mode_var).pack(side="left")
        ttk.Radiobutton(self.search_frame, text="行人", value="walking", variable=self.mode_var).pack(side="left")

        self.plan_btn = ttk.Button(self.search_frame, text="规划路径", command=self.plan_path)
        self.plan_btn.pack(side="left", padx=5)

        self.map_frame = ttk.Frame(self.map_root)
        self.map_frame.pack(fill="both", expand=True)
        ttk.Label(self.map_frame, text="地图将在浏览器中打开").pack(pady=10)

    def setup_video_gui(self):
        self.video_frame = ttk.Frame(self.video_root)
        self.video_frame.pack(fill="both", expand=True)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)

        self.control_frame = ttk.Frame(self.video_root)
        self.control_frame.pack(fill="x", pady=5)
        self.start_btn = ttk.Button(self.control_frame, text="启动检测", command=self.start_detection)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(self.control_frame, text="停止检测", command=self.stop_detection)
        self.stop_btn.pack(side="left", padx=5)
        self.status_label = ttk.Label(self.control_frame, text="状态：待机")
        self.status_label.pack(side="left", padx=5)

    def plan_path(self):
        start_loc = self.geolocator.geocode(self.start_entry.get())
        end_loc = self.geolocator.geocode(self.end_entry.get())

        if not start_loc or not end_loc:
            messagebox.showerror("错误", "无法找到起点或终点")
            return

        start = [start_loc.longitude, start_loc.latitude]
        end = [end_loc.longitude, end_loc.latitude]

        path = self.get_amap_path(start, end, self.mode_var.get())
        if not path:
            path = self.dwa_path_planning(start, end)
            messagebox.showwarning("警告", "高德API不可用，使用简化路径")

        m = folium.Map(location=[start[1], start[0]], zoom_start=14)
        folium.PolyLine(path, color="blue", weight=5).add_to(m)

        folium.Marker(
            location=[start[1], start[0]],
            popup="起点",
            icon=folium.Icon(color="blue", icon="map-pin", prefix="fa")
        ).add_to(m)
        folium.Marker(
            location=[end[1], end[0]],
            popup="终点",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa")
        ).add_to(m)

        map_file = "map.html"
        m.save(map_file)
        webbrowser.open(f"file://{os.path.abspath(map_file)}")

    def get_amap_path(self, start, end, mode):
        url = f"https://restapi.amap.com/v3/direction/{mode}?origin={start[0]},{start[1]}&destination={end[0]},{end[1]}&key={self.amap_api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            if data["status"] == "1" and data["route"]["paths"]:
                path_points = []
                for step in data["route"]["paths"][0]["steps"]:
                    for polyline in step["polyline"].split(";"):
                        lon, lat = map(float, polyline.split(","))
                        path_points.append([lat, lon])
                return path_points
        except:
            return None
        return None

    def dwa_path_planning(self, start, end):
        num_points = 10
        path = []
        for i in range(num_points + 1):
            t = i / num_points
            lon = start[0] + t * (end[0] - start[0])
            lat = start[1] + t * (end[1] - start[0])
            path.append([lat, lon])
        return path

    def start_detection(self):
        if not self.running:
            self.running = True
            self.status_label.config(text="状态：检测中")
            threading.Thread(target=self.detect_obstacles, daemon=True).start()
            self.video_root.after(100, self.update_video)

    def stop_detection(self):
        self.running = False
        self.status_label.config(text="状态：待机")

    def detect_obstacles(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot(labels=True, conf=True)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (600, 350))
            self.current_frame = frame_resized

    def update_video(self):
        if self.running and self.current_frame is not None:
            img = tk.PhotoImage(data=cv2.imencode('.ppm', self.current_frame)[1].tobytes())
            self.video_label.configure(image=img)
            self.video_label.image = img
        if self.running:
            self.video_root.after(100, self.update_video)

    def on_closing(self):
        self.running = False
        self.cap.release()
        if os.path.exists("map.html"):
            os.remove("map.html")
        self.video_root.destroy()
        self.map_root.destroy()


if __name__ == "__main__":
    app = AutonomousDrivingSim()
    app.map_root.mainloop()