import time
import tkinter as tk
from tkinter import Canvas ,filedialog
import sys
from PIL import Image,ImageTk
from rembg import remove
from deep_translator import GoogleTranslator
import torch
import imageio
from diffusers import DiffusionPipeline,StableVideoDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import cv2
import numpy as np
import os
import numpy as np
import os
from tqdm import tqdm

prompt_content=None
# Check if the prompt file exists and read its content
if os.path.exists(".//prompt.txt"):
    with open(".//prompt.txt", 'r') as rp:
        prompt_content = rp.read()
    if prompt_content:
        os.remove(".//prompt.txt")


path1 = None
path2 = None
path_sfondo = None

window = tk.Tk()
window.attributes('-fullscreen', True)
frame = Canvas(window, width=960, height=720, bg='red')
frame.pack()

# Create a frame for buttons
button_frame = tk.Frame(window)
button_frame.pack(side='top', fill='x')

def pathimage1():
    global path1
    file_path = filedialog.askopenfilename(title="Seleziona image_1")
    if file_path:  # Se un file è stato selezionato
        path1 = file_path
    else:  # Se l'utente ha annullato la selezione
        path1 = None
    print(f"path1 {path1}")

def pathimage2():
    global path2
    file_path = filedialog.askopenfilename(title="Seleziona image_2")
    if file_path:
        path2 = file_path
    else:
        path2 = None
    print(f"path2 {path2}")

def pathimage_sfondo():
    global path_sfondo
    file_path = filedialog.askopenfilename(title="Seleziona image_sfondo")
    if file_path:
        path_sfondo = file_path
    else:
        path_sfondo = None
    print(f"path_sfondo {path_sfondo}")
    
import os
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from deep_translator import GoogleTranslator
from tqdm import tqdm
import shutil

def texttovideo():
    global prompt, prompt_negativo
    if not os.path.exists('./collage.png'):
        print("Error: collage.png not found.")
        return

    image = Image.open("./collage.png")
    image.thumbnail((512, 512))  # Resize the image
    
    def traduci(txt):
        try:
            translator = GoogleTranslator(source='it', target='en')
            return translator.translate(txt)
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return txt  # Return original text if translation fails

    prompteng = traduci(prompt.get('1.0','end'))
    shutil.copyfile('./collage.png','.//PIA//example//collage.png')
    loop=False
    if 'loop' in prompteng:
        loop=True
        prompteng=prompteng.replace('loop','') 
    
    filecong=f"""base: 'example/config/base.yaml'
prompts:
- - {prompteng}
n_prompt:
  - '(Blurry), (duplicate), (Deformed), (octane render, render, drawing, souls, bad photo, bad photography: 1.3), (Worst Quality, Low Quality, Blurry: 1.2), (Bad Teeth, Deformed Teeth,Deformed Lips), (Bad Anatomy, Bad Proportions: 1.1), (Deformed Iris, Deviced Pupils), (Deformed Eyes, Bad Eyes), (Deformed Face, Ugly Face, Bad Face), (Deformed Hands, Bad Hands, Fused Fingers), soft, mutilated, mutation, dysfigured'
guidance_scale: 7.5
validation_data:
  input_name: 'collage'
  validation_input_path: '.\example'
  save_path: 'example/result'
  mask_sim_range: [-3]
generate:
  use_lora: false
  use_db: true
  global_seed: 5658137986800322011
  lora_path: ""
  db_path: "models/DreamBooth_LoRA/realisticVisionV60B1_v51HyperVAE.safetensors"
  lora_alpha: 1.0"""
  
    with open(".//PIA//example//config//video.yaml", "w") as f:
        f.write(filecong)
    if os.path.exists(".//PIA//example//config//video.yaml"):
        os.chdir('.//PIA')
        if loop== True:
            os.system("python inference.py --config=example/config/video.yaml --loop")
            loop=False
        else:
            os.system("python inference.py --config=example/config/video.yaml")
        time.sleep(1)
        os.chdir('..')
    else:
        print("file video.yaml non esiste")
        return
        

    

    
    

       
    
def creaimagine():
    global path1, path2, path_sfondo, frame
    imagecollage = Image.new('RGB', (960, 720), 'green')
    
    img1 = None
    if path1 is not None:
        img1 = Image.open(path1)
        if img1.mode != 'RGBA':
            img1 = remove(img1)
    
    img2 = None
    if path2 is not None:
        img2 = Image.open(path2)
        if img2.mode != 'RGBA':
            img2 = remove(img2)
    
    img_sfondo = None
    if path_sfondo is not None:
        img_sfondo = Image.open(path_sfondo)
    
    if img_sfondo is not None:
        img_sfondo = img_sfondo.resize((960, 720), Image.LANCZOS)
        imagecollage.paste(img_sfondo, (0, 0))
    
    if img1 is not None:
        w, h = img1.size
        x = (720 * w) // h
        img1 = img1.resize((x, 720), Image.LANCZOS)
        imagecollage.paste(img1, (0, 0), img1)
        
    if img2 is not None:
        w, h = img2.size
        x = (720 * w) // h
        img2 = img2.resize((x, 720), Image.LANCZOS)
        pos_offset=0
        pos_offset=imagecollage.width-img2.width-10
        imagecollage.paste(img2, (pos_offset, 0), img2)
    
    imagecollage.save("./collage.png")
    
    # Converti l'immagine PIL in un oggetto PhotoImage
    tk_image = ImageTk.PhotoImage(imagecollage)
    
    # Visualizza l'immagine nel canvas
    frame.create_image(0, 0, anchor='nw', image=tk_image)
    frame.image = tk_image  # Mantieni un riferimento all'immagine
    frame.update()
    time.sleep(2)
    texttovideo()

# Buttons aligned at the bottom
button1 = tk.Button(button_frame, text="image_1", command=pathimage1)
button1.pack(side='left', expand=True)
button2 = tk.Button(button_frame, text="image_2", command=pathimage2)
button2.pack(side='left', expand=True)
button3 = tk.Button(button_frame, text="image_sfondo", command=pathimage_sfondo)
button3.pack(side='left', expand=True)

# Create a frame for text widgets
text_frame = tk.Frame(window)
text_frame.pack(side='top', fill='x')

# Text widgets below the buttons, aligned
prompt = tk.Text(text_frame, height=7)
prompt.pack(side='left', expand=True, fill='x')
if not prompt_content==None:
    prompt.insert('1.0', prompt_content)  # Insert the content into the text widget
    
prompt_negativo = tk.Text(text_frame, height=7)
prompt_negativo.pack(side='left', expand=True, fill='x')

# Create a frame for Generate and Exit buttons
control_frame = tk.Frame(window)
control_frame.pack(side='top', fill='x', pady=10)

# Create an inner frame to hold the buttons
button_container = tk.Frame(control_frame)
button_container.pack(expand=True)

# Generate button
generate_button = tk.Button(button_container, text='Genera',command=creaimagine)
generate_button.pack(side='left', padx=5)

def ex():
    window.destroy()
    sys.exit()

# Exit button
exit_button = tk.Button(button_container, text='Exit', bg='red', command=ex)
exit_button.pack(side='left', padx=5)


window.mainloop()
