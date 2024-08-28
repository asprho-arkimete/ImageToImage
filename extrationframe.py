import tkinter as tk
from tkinter import filedialog, Canvas, Button, Scale,Label,messagebox
from tkinter.ttk import Combobox
from moviepy.editor import VideoFileClip
from PIL import Image, ImageTk
import numpy as np
import os
import os
from PIL import ImageGrab


video = None
pathfile = None
clips_list = []

window = tk.Tk()
window.geometry('700x700')
window.resizable(False, False)  # Disabilita il ridimensionamento della finestra

# Funzione per aggiornare i fotogrammi nella Canvas
def update_frame(event=None):
    global video, clips, sc, frame,frame_number,labal
    if video is not None:
        selected_clip = clips.current()
        if selected_clip != -1:
            start_frame = selected_clip * 100
            end_frame = min(start_frame + 100, int(video.duration * video.fps))
            frame_number = start_frame + sc.get()
            if frame_number < end_frame:
                img = video.get_frame(frame_number / video.fps)
                # Convert numpy array to PIL Image
                img = Image.fromarray(img)
                # Resize the image to fit the Canvas
                w, h = img.size
                
                if w >= h:
                    X = (512 * h) // w
                    img = img.resize((512, X), Image.LANCZOS)
                else:
                    X = (512 * w) // h
                    img = img.resize((X, 512), Image.LANCZOS)
                
                img = ImageTk.PhotoImage(img)
                # Calculate position to center the image
                canvas_width = frame.winfo_width()
                canvas_height = frame.winfo_height()
                x_pos = (canvas_width - img.width()) // 2
                y_pos = (canvas_height - img.height()) // 2
                
                frame.create_image(x_pos, y_pos, anchor=tk.NW, image=img)
                frame.image = img
                frame.update()
                labal.config(text=f"Fotogramma: {frame_number}")
                labal.update()
                

# Funzione per gestire il cambiamento di selezione nella combobox
def on_clip_change(event=None):
    sc.set(0)  # Reset the scale value to 0
    update_frame()

# Funzione per dividere il video in clip da 100 fotogrammi
def clipF(event=None):
    global video, clips, sc, clips_list
    clips_list = []
    if video is not None:
        total_frames = int(video.duration * video.fps)
        for i in range(0, total_frames, 100):
            clips_list.append(f"clip {i // 100 + 1}")
    clips.config(values=clips_list)
    clips.current(0)
    sc.config(to=100)
    update_frame()

def load():
    global video, pathfile
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mkv"),
        ("All files", "*.*")
    ]
    pathfile = filedialog.askopenfilename(filetypes=filetypes)
    if pathfile:
        video = VideoFileClip(pathfile)
        clipF()

button_load = Button(window, text='Load video', command=load)
button_load.pack(side='top')

frame = Canvas(window, width=512, height=512,bg='white')
frame.pack(side='top')

sc = Scale(window, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, command=update_frame)
sc.pack(side='top')

clips = Combobox(window)
clips.pack(side='top')
clips.bind("<<ComboboxSelected>>", on_clip_change)

def saveframe():
    global frame, pathfile, frame_number, video
    nome = os.path.basename(pathfile).split('.')[0]
    img = video.get_frame(frame_number / video.fps)
    # Convert numpy array to PIL Image
    img = Image.fromarray(img)
    # Save the image
    img.save(f'fotogramma_{nome}_{frame_number}.png', format='PNG')
    messagebox.showinfo("Frame Saved", f"Frame {frame_number} saved successfully", icon=messagebox.INFO)
    

button_save = Button(window, text="Salva fotogramma", command=saveframe)
button_save.pack(side='top')

labal= Label(window,text="Nessun Fotogramma")
labal.pack(side='top')

window.mainloop()