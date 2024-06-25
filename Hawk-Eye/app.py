from PIL import ImageTk, Image
import customtkinter as tk
import threading
import queue
import time
import datetime
import numpy as np
import cv2
import os
import subprocess
import base64
import requests
import json
import pygame
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler
from torch.utils.data import DataLoader
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute
)
from torchvision.transforms import (
    Compose,
    Lambda,
    CenterCrop,
    RandomHorizontalFlip
)
import tkinter.messagebox as messagebox

server_url=''
server_port=''

# Initialize pygame for sound playback
pygame.mixer.init()

# Load the alert sound
alert_sound_path = "alert_sound.mp3"  # Path to your alert sound file
if os.path.exists(alert_sound_path):
    alert_sound = pygame.mixer.Sound(alert_sound_path)
else:
    alert_sound = None
    print("Alert sound file not found!")

# Define constants
HIGH_PROB_THRESHOLD = 0.8
MEDIUM_PROB_THRESHOLD = 0.6
VIDEO_WEIGHT = 0.8
FRAMES_TO_ANALYZE = 1
FRAME_FREQUENCY = 0.1
BATCH_SIZE = 20
UI_COLORS = ['#48ff00', '#f6ff00', '#ff0000']
ALERT_LENGTH = 20
QUEUE_MAXSIZE = 10
V_DETECTED = False

# Define video transform
video_transform = Compose([
    ApplyTransformToKey(key='video',
        transform=Compose([
            UniformTemporalSubsample(20),
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            RandomShortSideScale(min_size=248, max_size=256),
            CenterCrop(224),
            RandomHorizontalFlip(p=0.5)
        ])
    ),
])

# Define model
class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()
        self.video_model = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(400, 1)
        self.lr = 1e-3
        self.metric = torchmetrics.Accuracy(task="binary")
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.video_model(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        dataset = labeled_video_dataset(train_df, clip_sampler=make_clip_sampler('random', 2), transform=video_transform, decode_audio=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader

    def training_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out = self(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {"loss": loss, "metric": metric.detach()}

    def val_dataloader(self):
        dataset = labeled_video_dataset(val_df, clip_sampler=make_clip_sampler('random', 2), transform=video_transform, decode_audio=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.numworker, pin_memory=True)
        return loader

    def validation_step(self, batch, batch_idx):
        video, label = batch['video'], batch['label']
        out = self(video)
        loss = self.criterion(out, label)
        metric = self.metric(out, label.to(torch.int64))
        return {"loss": loss, "metric": metric.detach()}

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OurModel().to(device)
checkpoint_path = 'last.ckpt'
model = OurModel.load_from_checkpoint(checkpoint_path).to(device)

# Global variables
STOP_FLAG = False
frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
output_queue = queue.Queue()

def capture_frames(camera_index):
    global STOP_FLAG
    camera_index = int(camera_index)
    cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
    while not STOP_FLAG:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from the video stream.")
            continue
        if not frame_queue.full():
            frame_queue.put(frame)
        time.sleep(FRAME_FREQUENCY)
    cap.release()

def save_video_and_send(output_path, alert_validity, camera_ip):
    batch_frames = []
    try:
        print("Starting to collect frames...")
        while len(batch_frames) <= ALERT_LENGTH:
            frame = frame_queue.get(timeout=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
        print(f"Collected {len(batch_frames)} frames.")
        
        if not batch_frames:
            print("Error: No frames to save.")
            return
        
        # Get current date for the incident
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        print(f"Current date for the incident: {current_date}")
        
        # Create the directory if it doesn't exist
        logs_directory = os.path.join(os.getcwd(), "logs", current_date)
        os.makedirs(logs_directory, exist_ok=True)
        print(f"Logs directory created or exists: {logs_directory}")
        
        # Set the output path within the logs directory
        output_path = os.path.join(logs_directory, output_path)
        print(f"Output path set: {output_path}")
        
        # Write frames to video file
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = batch_frames[0].shape
        out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (width, height))
        print(f"Writing frames to temporary video file: {temp_video_path}")
        for frame in batch_frames:
            out.write(frame)
        out.release()  # Release the video writer
        print("Finished writing frames to temporary video file.")
        
        # Convert to browser-supported format using ffmpeg
        ffmpeg_command = [
            'ffmpeg', '-i', temp_video_path, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-b:a', '192k', output_path
        ]
        print(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
        print("Finished converting video with ffmpeg.")
        
        # Remove the temporary video file
        os.remove(temp_video_path)
        print(f"Removed temporary video file: {temp_video_path}")
        
        # Prepare JSON data
        current_datetime = datetime.datetime.now()
        json_data = {
            'date': {
                'day': current_datetime.strftime("%A"),  # Get the full day name (e.g., Monday)
                'time': current_datetime.strftime("%H:%M:%S"),  # Get the time in HH:MM:SS format
            },
            'camera_ip': camera_ip,
            'alert_validity': alert_validity
        }
        print(f"Prepared JSON data: {json_data}")
        
        # Read video file as binary and encode to base64
        with open(output_path, 'rb') as video_file:
            binary_data = video_file.read()
            base64_data = base64.b64encode(binary_data)
        print(f"Encoded video file to base64.")
        
        # Prepare payload
        payload = {
            'json_data': json.dumps(json_data),
            'video_blob': base64_data.decode('utf-8')  # Decode bytes to string
        }
        print("Prepared payload for server.")
        
        # Send data to server
        server_url = '192.168.8.131'  # Replace with the server URL or IP address
        server_port = '5001'  # Replace with the port the server is listening on
        print(f"Sending data to server http://{server_url}:{server_port}/office/upload")
        response = requests.post(f'http://{server_url}:{server_port}/office/upload', data=payload)
        
        # Check response status and print message
        if response.status_code == 200:
            print("Video and JSON data uploaded successfully.")
        else:
            print(f"Failed to upload video and JSON data. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


def video_analysis(batch_size=BATCH_SIZE):
    global STOP_FLAG
    def get_probabilities_for_batch(frames):
        frames_array = np.array(frames, dtype=np.float32)
        video_tensor = torch.tensor(frames_array, dtype=torch.float32).permute(3, 0, 1, 2).to(device)
        video_data = {'video': video_tensor}
        video_data = video_transform(video_data)
        pred = model(video_data['video'].unsqueeze(0).to(device))
        probability = torch.sigmoid(pred).item()
        print("probability:", probability)
        return [probability]

    last_scores = []
    batch_frames = []
    while not STOP_FLAG:
        try:
            frame = frame_queue.get(timeout=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            if len(batch_frames) >= batch_size:
                start_time = time.time()
                probs = get_probabilities_for_batch(batch_frames)
                end_time = time.time()
                time_interval = end_time - start_time
                last_scores.extend(probs)
                batch_frames = []
                while len(last_scores) > FRAMES_TO_ANALYZE:
                    last_scores = last_scores[1:]
                final_score = max(last_scores)
                output_queue.put((0, final_score))
        except queue.Empty:
            continue
        except Exception as e:
            print(f"An error occurred: {e}")

class Application(tk.CTk):
    def __init__(self, master=None, camera_index=0, server_url='', server_port=''):
        super().__init__(master)
        self.final_prob = 0
        self.camera_index = camera_index
        self.last_probs = [None, None]
        self.master = master
        self.camera_frames = []
        self.camera_labels = []
        self.report_buttons = []
        self.decline_buttons = []
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.server = server_url
        self.port = server_port
        self.declined = False  # Add this flag
        self.create_widgets()


    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        global server_url
        server_url = self.server
        global server_port
        server_port = self.port

        # Create canvas
        self.canvas = tk.CTkCanvas(self, width=300, height=5, bg='black', highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)

        for i in range(4):
            frame = tk.CTkFrame(self)
            frame.grid(row=i//2 + 1, column=i%2, padx=5, pady=5, sticky="nsew")
            self.camera_frames.append(frame)
            
            label = tk.CTkLabel(frame, text='')
            label.grid(row=0, column=0, sticky="nsew")
            self.camera_labels.append(label)

            # Report button
            report_button = tk.CTkButton(frame, text="Report", command=self.report)
            report_button.grid(row=1, column=0, pady=(0, 10), padx=(0, 300))
            self.report_buttons.append(report_button)
            # Decline button
            decline_button = tk.CTkButton(frame, text="Decline", command=self.decline, fg_color='#7D1012', hover_color='#DD1417')
            decline_button.grid(row=1, column=0, pady=(0, 10), padx=(300, 0))
            self.decline_buttons.append(decline_button)

        # Load the static image and split it into 4 parts
        placeholder_path = "No_Signals.jpg"
        if os.path.exists(placeholder_path):
            static_img = Image.open(placeholder_path)
            static_img = static_img.resize((640, 360), Image.LANCZOS)
            for i in range(4):
                part_img_tk = tk.CTkImage(static_img, size=(710, 360))
                self.camera_labels[i].configure(image=part_img_tk)
                self.camera_labels[i].image = part_img_tk
        else:
            print("AddCamera image not found!")

        self.feedback_button = tk.CTkButton(self, text="Feedback", command=self.open_feedback_window, fg_color='#D79023', hover_color='#F0AB45')
        self.feedback_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.start_analysis()
        self.update_ui()

    def open_feedback_window(self):
        feedback_window = FeedbackWindow()
        feedback_window.title("Feedback")
        feedback_window.geometry("600x150")
        feedback_window.resizable(False, False)
        feedback_window.mainloop()

    def report(self):
        global V_DETECTED
        video_path = f"violence_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        if V_DETECTED:
            print("detected")
            save_thread = threading.Thread(target=save_video_and_send, args=(video_path, 'TP', '10.190.18.40',))
            save_thread.start()
        else:
            print("not detected")
            save_thread = threading.Thread(target=save_video_and_send, args=(video_path, 'FN', '10.190.18.40',))
            save_thread.start()

    def decline(self):
        global V_DETECTED
        V_DETECTED = False
        self.declined = True  # Set the declined flag
        video_path = f"violence_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        if V_DETECTED:
            save_thread = threading.Thread(target=save_video_and_send, args=(video_path, 'FP', '10:190:18:40',))
            save_thread.start()
        
        # Stop the alert sound if it's playing
        if alert_sound and pygame.mixer.get_busy():
            alert_sound.stop()
        
        # Reset the UI color to default
        self.canvas.config(background=UI_COLORS[0])


    def start_analysis(self):
        self.t1 = threading.Thread(target=capture_frames, args=(self.camera_index,))
        self.t2 = threading.Thread(target=video_analysis)
        self.t1.start()
        self.t2.start()

    def update_color(self):
        global V_DETECTED
        if self.final_prob >= HIGH_PROB_THRESHOLD:
            next_color = UI_COLORS[2]
            V_DETECTED = True
            if not self.declined and alert_sound and not pygame.mixer.get_busy():
                alert_sound.play(-1)  # Loop the sound
        elif self.final_prob >= MEDIUM_PROB_THRESHOLD:
            V_DETECTED = True
            next_color = UI_COLORS[1]
            if alert_sound and pygame.mixer.get_busy():
                alert_sound.stop()  # Stop the sound
        else:
            V_DETECTED = False
            next_color = UI_COLORS[0]
            self.canvas.delete("all")
            if alert_sound and pygame.mixer.get_busy():
                alert_sound.stop()  # Stop the sound

        # Reset the declined flag if final_prob falls below MEDIUM_PROB_THRESHOLD
        if self.final_prob < MEDIUM_PROB_THRESHOLD:
            self.declined = False

        self.canvas.config(background=next_color)

    def update_ui(self):
        try:
            output = output_queue.get_nowait()
            self.last_probs[output[0]] = output[1]
        except queue.Empty:
            pass

        self.final_prob = 0
        if self.last_probs[0] is not None and self.last_probs[0] >= 0:
            if self.last_probs[1] is not None and self.last_probs[1] >= 0:
                self.final_prob = VIDEO_WEIGHT * self.last_probs[0] + (1 - VIDEO_WEIGHT) * self.last_probs[1]
            if self.last_probs[1] is None or self.last_probs[1] < 0:
                self.final_prob = self.last_probs[0]
        elif self.last_probs[1] is not None and self.last_probs[1] >= 0:
            self.final_prob = self.last_probs[1]

        try:
            frame = frame_queue.get(timeout=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((640, 360), Image.LANCZOS)
            frame_tk = tk.CTkImage(frame, size=(710, 360))
            self.camera_labels[0].configure(image=frame_tk)
            self.camera_labels[0].image = frame_tk

            self.update_color() 
        except queue.Empty:
            pass

        self.after(30, self.update_ui)

    def on_close(self):
        global STOP_FLAG
        STOP_FLAG = True
        self.destroy()
        if hasattr(self, 't1') and isinstance(self.t1, threading.Thread) and self.t1.is_alive():
            self.t1.join()
        if hasattr(self, 't2') and isinstance(self.t2, threading.Thread) and self.t2.is_alive():
            self.t2.join()


class FeedbackWindow(tk.CTk):
    def __init__(self, master=None):
        super().__init__(master)
        self.feedback_label = tk.CTkLabel(self, text="Please provide your feedback:")
        self.feedback_label.grid(row=0, column=0, padx=10, pady=5)

        self.feedback_entry = tk.CTkEntry(self,width=500)
        self.feedback_entry.grid(row=1, column=0, padx=30, pady=5)

        self.submit_button = tk.CTkButton(self, text="Submit", command=self.submit_feedback)
        self.submit_button.grid(row=2, column=0, padx=10, pady=5)

    def submit_feedback(self):
        feedback_text = self.feedback_entry.get()
        current_time_ms = int(datetime.datetime.now().timestamp() * 1000)
        # Process the feedback (e.g., send it to a server, save to a file, etc.)
        payload = {
            'date': current_time_ms,
            'report_description': feedback_text,
            "reported_personnel": "Nati"
        }
        print("[THE REPORT]", payload)
        global server_url
        global server_port
        
        try:
            print(f"Sending report to server http://{server_url}:{server_port}/office/failure_report")
            response = requests.post(f'http://{server_url}:{server_port}/office/failure_report', data=payload,timeout=3)
            if response.status_code == 200:
                print("Report data uploaded successfully.")
            else:
                print(f"Failed to upload Report data. Status code: {response.status_code}")
            self.destroy()
        except requests.Timeout:
            self.destroy()
            messagebox.showerror("Timeout occurred","Server did not respond within the specified time.")
            
        except requests.RequestException as e:
            self.destroy()
            messagebox.showerror("Error","Request Exception occurred.")
            
        except Exception as e:
            self.destroy()
            messagebox.showerror("Error","Server may not be up.")
            
class LoginWindow(tk.CTk):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.title("Login")
        self.geometry("600x350")  # Increased dimensions for the login window
        # Add padding and empty rows to move everything down
        self.padding_label = tk.CTkLabel(self, text="", height=2)
        self.padding_label.grid(row=0, column=0)
        self.empty_row_label = tk.CTkLabel(self, text="", height=1)
        self.empty_row_label.grid(row=1, column=0)
        
        self.username_label = tk.CTkLabel(self, text="Username:")
        self.username_label.grid(row=2, column=0, padx=10, pady=5)
        self.username_entry = tk.CTkEntry(self)
        self.username_entry.grid(row=2, column=1, padx=10, pady=5)
        
        self.password_label = tk.CTkLabel(self, text="Password:")
        self.password_label.grid(row=3, column=0, padx=10, pady=5)
        self.password_entry = tk.CTkEntry(self, show="*")
        self.password_entry.grid(row=3, column=1, padx=10, pady=5)
        
        self.camera_index_label = tk.CTkLabel(self, text="Camera :")
        self.camera_index_label.grid(row=4, column=0, padx=10, pady=5)
        self.camera_index_entry = tk.CTkEntry(self)
        self.camera_index_entry.grid(row=4, column=1, padx=10, pady=5)
        
        self.server_url_label = tk.CTkLabel(self, text="Server URL:")
        self.server_url_label.grid(row=5, column=0, padx=10, pady=5)
        self.server_url_entry = tk.CTkEntry(self)
        self.server_url_entry.grid(row=5, column=1, padx=10, pady=5)
        
        self.server_port_label = tk.CTkLabel(self, text="Server Port:")
        self.server_port_label.grid(row=6, column=0, padx=10, pady=5)
        self.server_port_entry = tk.CTkEntry(self)
        self.server_port_entry.grid(row=6, column=1, padx=10, pady=5)
        
        self.empty_row_label = tk.CTkLabel(self, text="", height=1)
        self.empty_row_label.grid(row=7, column=0)
        
        self.login_button = tk.CTkButton(self, text="Login", command=self.login)
        self.login_button.grid(row=8, column=0, columnspan=2, padx=225, pady=15)

    def login(self):
        # Check username, password, camera index, server URL, and server port (replace with your authentication logic)
        username = self.username_entry.get()
        password = self.password_entry.get()
        camera_index = self.camera_index_entry.get()  # Get camera index
        server_url = self.server_url_entry.get()  # Get server URL
        server_port = self.server_port_entry.get()  # Get server port
        
        if not server_url or not server_port:
            server_url = '127.0.0.1'
            server_port = '5001'
        if not camera_index:
            messagebox.showerror("Error", "Camera index cannot be empty.")
            return
        if username == "admin" and password == "admin":
            self.destroy()
            app = Application(camera_index=camera_index, server_url=server_url, server_port=server_port)  # Pass server URL and port to Application
            icon1_path ='icon.ico'
            app.iconbitmap(icon1_path)
            app.title("Hawk-Eye")
            app.resizable(False, False)
            app.mainloop()
        else:
            messagebox.showerror("Error", "Invalid username or password.")

if __name__ == "__main__":
    login_window = LoginWindow()
    icon_path = 'icon.ico'
    login_window.iconbitmap(icon_path)
    login_window.mainloop()
