import gc
import os
import time
from tkinter import END, Tk, Frame, Canvas, Label, Text, Button,Scale,HORIZONTAL, filedialog
from tkinter.ttk import Combobox
from PIL import Image, ImageTk,ImageChops
from scipy import ndimage
from tkinterdnd2 import DND_FILES, TkinterDnD
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
from deep_translator import GoogleTranslator
from compel import Compel
import cv2
import dlib
import numpy as np
import shutil
from clip_interrogator import Config, Interrogator, LabelTable, load_list
from PIL import Image
import math
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,DPMSolverMultistepScheduler,StableDiffusionXLImg2ImgPipeline
from PIL import ImageGrab,ImageOps,ImageDraw
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
import torch
from skimage import filters, morphology
from rembg import remove
import requests
from bs4 import BeautifulSoup
import webbrowser
from google.cloud import vision

#estrai fotogrammi lib
import tkinter as tk
from tkinter import filedialog, Canvas, Button, Scale,Label,messagebox
from tkinter.ttk import Combobox
from moviepy.editor import VideoFileClip
from PIL import Image, ImageTk
import numpy as np
import os
import os
from PIL import ImageGrab

#texttovideo lib
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
 

if os.path.exists(".//mask_final.png"):
    os.remove(".//mask_final.png")
if os.path.exists(".//mask_penna.png"):
    os.remove(".//mask_penna.png") 

Poseactivate=False
 # Create Config object
config = Config(clip_model_name="ViT-L-14/openai")
    
# Create Interrogator with the config
ci = Interrogator(config)

def clip_interogation():
    global img, text,ci
    if img is not None:
        try:
            # Convert image to RGB
            image = img.convert('RGB')
            # Get the English description
            description = ci.interrogate(image)
            print(description)
            try:
                # Translate to Italian
                translator = GoogleTranslator(source='en', target='it')
                traduzione = translator.translate(description)
                
                # Clear previous text and insert new translation
                text.delete('1.0', 'end')
                text.insert('1.0', traduzione)
            except Exception as e:
                error_message = f"Errore durante la traduzione: {str(e)}"
                print(error_message)
                
                # Display error message in the text widget
                text.delete('1.0', 'end')
                text.insert('1.0', error_message)
        except Exception as errorImg:
            print(f"nessuna Immagine Input: {errorImg}")
    else:
        print("nessuna immagine Input")
    
def drop_image(event, canvas):
    global img,text
    file_path = event.data
    if file_path:
        img = Image.open(file_path)
        img.save(".//temp_image.png")
        print("salvo immagine temp_ imagedrop")
        
        # Calcola le nuove dimensioni mantenendo le proporzioni
        canvas_ratio = canvas.winfo_width() / canvas.winfo_height()
        img_ratio = img.width / img.height
        
        if img_ratio > canvas_ratio:
            new_width = canvas.winfo_width()
            new_height = int(new_width / img_ratio)
        else:
            new_height = canvas.winfo_height()
            new_width = int(new_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        # Centra l'immagine nel canvas
        x = (canvas.winfo_width() - new_width) // 2
        y = (canvas.winfo_height() - new_height) // 2
        
        canvas.delete("all")  # Rimuove l'immagine precedente
        canvas.create_image(x, y, anchor='nw', image=img_tk)
        canvas.image = img_tk  # Mantiene un riferimento per evitare la garbage collection
        # Aggiorna il canvas prima di chiamare after
        canvas.update_idletasks()
        
        
        
        
        if text.get('1.0', 'end-1c') == '':
                canvas.after(100, lambda: clip_interogation())
                
        
def drop_image_face(event, canvas):
    file_path = event.data
    if file_path:
        img = Image.open(file_path)
        img.save(".//temp_face.png")
        print("SALVO IMAGE TEMP VISO")
        
        # Calcola le nuove dimensioni mantenendo le proporzioni
        canvas_ratio = canvas.winfo_width() / canvas.winfo_height()
        img_ratio = img.width / img.height
        
        if img_ratio > canvas_ratio:
            new_width = canvas.winfo_width()
            new_height = int(new_width / img_ratio)
        else:
            new_height = canvas.winfo_height()
            new_width = int(new_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        # Centra l'immagine nel canvas
        x = (canvas.winfo_width() - new_width) // 2
        y = (canvas.winfo_height() - new_height) // 2
        
        canvas.delete("all")  # Rimuove l'immagine precedente
        canvas.create_image(x, y, anchor='nw', image=img_tk)
        canvas.image = img_tk  # Mantiene un riferimento per evitare la garbage collection
        
def traduci_ita_eng(testo):
    try:
        translator = GoogleTranslator(source='it', target='en')
        traduzione = translator.translate(testo)
        return traduzione
    except Exception as e:
        return f"Errore durante la traduzione: {str(e)}"
t=1    

def refine_f(p):
    global model, text, textneg, t
    print("REFINE")
    
    try:
        # Clean CUDA cache and free unused memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the pipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        ).to(device)
        
        init_image = Image.open(p).convert("RGB")
        
        x,y=(960,720)
        x,y= init_image.size
        
        prompt = traduci_ita_eng(text.get("1.0", "end-1c"))
        negative_prompt = traduci_ita_eng(textneg.get("1.0", "end-1c"))
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            width=x,
            height=y,
            
        ).images[0]
        
        image.save(f".//temp//Imagerefine_{t}.png")
        image.save(".//Imagerefine.png")
        
        # Clean up memory after use
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Refinement completed successfully")
    
    except Exception as e:
        print(f"An error occurred during refinement: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure memory is cleaned even in case of error
        torch.cuda.empty_cache()
        gc.collect()
        
maskesistente= False
def imgtoimag():
    global text, textneg,model, risoluzione, frame1, img_tk, CFG, STEPS, strength, eta,Poseactivate,framepose,DPM,funzione,maskesistente,t,refine,fineturring
    if funzione.get()== 'Estrai Fotogrammi':
        os.system("python extrationframe.py")
        return
    if funzione.get() == 'Video':
        if not text.get("1.0", "end-1c") == '':
            with open('.//prompt.txt', 'w') as fp:  # Open file in text mode
                fp.write(text.get("1.0", "end-1c"))
            os.system('python texttovideo.py')
        # Exit the function
        return
   
        
    # Pulisci la cache CUDA e libera la memoria non utilizzata
    torch.cuda.empty_cache()
    gc.collect()
    os.makedirs('.//temp',exist_ok=True)
   
    modelid = "digiplay/AnalogMadness-realistic-model-v7"
    
    model7inp= ".\\model_v7_inpaint\\analogMadness_v70-inpainting.safetensors"
    model7= "digiplay/AnalogMadness-realistic-model-v7"
    
    realistv6_b1_inpainting="stablediffusionapi/realistic-vision-v6.0-b1-inpaint"
    modelrealvision = "SG161222/Realistic_Vision_V6.0_B1_noVAE" 
    
    p_id_inp=".\\model_v7_inpaint\\uberRealisticPornMerge_urpmv13Inpainting.safetensors"
    P_id = "stablediffusionapi/uber-realistic-merge"
    P_id2 = "ductridev/uber-realistic-porn-merge-urpm"
    
    # Get values from the new sliders
    cfg_value = CFG.get()
    steps_value = STEPS.get()
    strength_value = strength.get()
    eta_value = eta.get()
    
    if model.get() == 'model_v7' and funzione.get()=='Rileva Vestiti':
        modelid= model7inp
    elif model.get() == 'model_v7':
        modelid = model7
    elif model.get() == 'realvisxlv60-B1' and funzione.get()=='Rileva Vestiti':
        modelid= realistv6_b1_inpainting
    elif model.get() == 'realvisxlv60-B1':
        modelid = modelrealvision
        
    elif model.get() == 'Rocco_v1' and funzione.get()=='Rileva Vestiti':
        modelid= p_id_inp
    elif model.get() == 'Rocco_v1':
        modelid = P_id
    elif model.get() == 'Rocco_v2' and funzione.get()=='Rileva Vestiti':
        modelid=p_id_inp
    elif model.get() == 'Rocco_v2':
        modelid = P_id2
    #[, 'Rileva Vestiti', 'Genera Immagini']
    if funzione.get()== 'ImageToImage':
        if Poseactivate==False:
            if model.get() == 'model_v7':
                pipeline = AutoPipelineForImage2Image.from_pretrained(modelid, torch_dtype=torch.float16, use_safetensors=True)
            else:
                pipeline = AutoPipelineForImage2Image.from_pretrained(modelid, torch_dtype=torch.float16)
        
        
        elif Poseactivate== True:
            openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            
            # Ottieni le coordinate del canvas
            x = framepose.winfo_rootx()
            y = framepose.winfo_rooty()
            x1 = x + framepose.winfo_width()
            y1 = y + framepose.winfo_height()
            # Cattura l'immagine del canvas
            ImageGrab.grab().crop((x, y, x1, y1)).save(".//image_pose.png")
            time.sleep(1)
            original_image = Image.open(".//image_pose.png")
            poseimage = openpose(original_image)
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(modelid, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    elif funzione.get()== 'Genera Immagini' and fineturring.get()=='No Fine-Tuning':
        if Poseactivate== True:
            openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            # Ottieni le coordinate del canvas
            x = framepose.winfo_rootx()
            y = framepose.winfo_rooty()
            x1 = x + framepose.winfo_width()
            y1 = y + framepose.winfo_height()
            # Cattura l'immagine del canvas
            ImageGrab.grab().crop((x, y, x1, y1)).save(".//image_pose.png")
            time.sleep(1)
            original_image = Image.open(".//image_pose.png")
            poseimage = openpose(original_image)
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(modelid, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            print(f"pose: {Poseactivate}")
        if model.get() == 'model_v7':
            pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
        else:
            pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16)
            
    elif funzione.get()=='Rileva Vestiti':
        if maskesistente== False:
            try:
                processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
                modello = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
                path = ".//temp_image.png"

                image = Image.open(path)
                inputs = processor(images=image, return_tensors="pt")

                outputs = modello(**inputs)
                logits = outputs.logits.cpu()

                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                

                pred_seg = upsampled_logits.argmax(dim=1)[0]

                # Create a mask for clothing items (labels 4, 5, 6, 7, 8)
                clothing_labels = [4, 5, 6, 7, 8]
                mask = torch.zeros_like(pred_seg)
                for label in clothing_labels:
                    mask = mask | (pred_seg == label)

                # Convert tensor to PIL image
                mask_image = Image.fromarray(mask.byte().numpy() * 255)

                # Save the image
                mask_image.save(".//mask.jpg")
                print("vestiti rivelati")
            except Exception as error:
                print(f"errore: {error}")
        #usa control net pose nel inpainting
        if Poseactivate== True:
            openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            
            # Ottieni le coordinate del canvas
            x = framepose.winfo_rootx()
            y = framepose.winfo_rooty()
            x1 = x + framepose.winfo_width()
            y1 = y + framepose.winfo_height()
            # Cattura l'immagine del canvas
            ImageGrab.grab().crop((x, y, x1, y1)).save(".//image_pose.png")
            time.sleep(1)
            original_image = Image.open(".//image_pose.png")
            poseimage = openpose(original_image)
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(modelid, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        
        if model.get() == 'model_v7':
            pipeline = StableDiffusionInpaintPipeline.from_single_file(modelid, torch_dtype=torch.float16,use_safetensors=True)
        elif model.get()=='realvisxlv60-B1':
            pipeline = AutoPipelineForInpainting.from_pretrained(modelid, torch_dtype=torch.float16)
        elif model.get()== 'Rocco_v1' or model.get()== 'Rocco_v2':
            pipeline = StableDiffusionInpaintPipeline.from_single_file(modelid, torch_dtype=torch.float16)
    elif funzione.get()=='Cambia Sfondo':
        #maschera sfondo
        input_path = './/temp_image.png'
        output_path = './/mask_final.png'
        # Rimuovi lo sfondo
        input_image = Image.open(input_path)
        output = remove(input_image)
        # Converti l'immagine in un array numpy
        output_array = np.array(output)

        # Crea una maschera dove il canale alpha è 0 (completamente trasparente)
        alpha_channel = output_array[:,:,3]
        mask = alpha_channel == 0

        # Crea un'immagine in bianco e nero
        result = np.zeros((output_array.shape[0], output_array.shape[1]), dtype=np.uint8)
        result[mask] = 255  # Imposta i pixel trasparenti a bianco

        # Converti la maschera in formato compatibile con OpenCV
        result_cv = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Applica una dilatazione per estendere la maschera bianca
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(result_cv, kernel, iterations=2)

        # Converti di nuovo in scala di grigi
        dilated_mask_gray = cv2.cvtColor(dilated_mask, cv2.COLOR_BGR2GRAY)

        # Salva l'immagine risultante
        Image.fromarray(dilated_mask_gray).save(output_path)

        print(f"Maschera estesa salvata come {output_path}")
        
        #inpainting
        if model.get() == 'model_v7':
            pipeline = StableDiffusionInpaintPipeline.from_single_file(modelid, torch_dtype=torch.float16,use_safetensors=True)
        elif model.get()=='realvisxlv60-B1':
            pipeline = AutoPipelineForInpainting.from_pretrained(modelid, torch_dtype=torch.float16)
        elif model.get()== 'Rocco_v1' or model.get()== 'Rocco_v2':
            pipeline = StableDiffusionInpaintPipeline.from_single_file(modelid, torch_dtype=torch.float16)
    
    
    
    
    if DPM.get()=='DPM' and not funzione.get()=='Rileva Vestiti' and fineturring.get()=='No Fine-Tuning':
        # Imposta il campionatore DPM++ SDE Karras
        scheduler = DPMSolverMultistepScheduler.from_pretrained(modelid, subfolder="scheduler")
        pipeline.scheduler = scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler.set_timesteps(num_inference_steps=steps_value)  # Numero di passi di inferenza
    
    if fineturring.get()=='No Fine-Tuning': 
        pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        #pipeline.enable_model_cpu_offload()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.to("cuda")
        prompt = traduci_ita_eng(text.get("1.0", "end-1c"))
        compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
        conditioning = compel(prompt)
        negative_prompt = traduci_ita_eng(textneg.get("1.0", "end-1c"))
        negative_conditioning = compel(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
    init_image = None
    x, y = 960, 720

    # prepare image
    if os.path.exists("./temp_image.png"):
        try:
            init_image = Image.open("./temp_image.png")
        except IOError:
            print("Impossibile aprire l'immagine temp_image.png")

    if risoluzione.get() == 'image Input' and init_image is not None and (funzione.get() == 'ImageToImage' or funzione.get() == 'Rileva Vestiti' or funzione.get() == 'Cambia Sfondo'):
        x, y = init_image.size
    elif not risoluzione.get() == 'image Input':
        x, y = map(int, risoluzione.get().split(','))


    # Ensure dimensions are always divisible by 8
    x = (x // 8) * 8
    y = (y // 8) * 8

    # Controllo aggiuntivo prima di accedere a init_image.size
    if init_image is not None:
        if init_image.size != (x, y):
            # Fai qualcosa qui, ad esempio ridimensiona l'immagine
            init_image = init_image.resize((x, y))
        else:
            print("Nessuna immagine iniziale disponibile")

    if funzione.get() == 'ImageToImage':
        if Poseactivate == True:
            # pass prompt and image to pipeline with new parameters
            image = pipeline(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                image=init_image,
                width=x,
                height=y,
                guidance_scale=cfg_value,
                num_inference_steps=steps_value,
                strength=strength_value,
                eta=eta_value,
                controlnetpose=poseimage
            ).images[0]
            image.save(f".//temp//imagegenerate_{t}.png")
            image.save(f".//imagegenerate.png")
            if (refine.get()== 'Refine Image'):
                refine_f(f".//imagegenerate.png")
            
            t=t+1
            if t>=10:
                t=1
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
        else:
            # pass prompt and image to pipeline with new parameters
            image = pipeline(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                image=init_image,
                width=x,
                height=y,
                guidance_scale=cfg_value,
                num_inference_steps=steps_value,
                strength=strength_value,
                eta=eta_value
            ).images[0]
            image.save(f".//temp//imagegenerate_{t}.png")
            image.save(f".//imagegenerate.png")
            if (refine.get()== 'Refine Image'):
                refine_f(f".//imagegenerate.png")
            
            t=t+1
            if t>=10:
                t=1
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
    elif funzione.get() == 'Genera Immagini':
        if fineturring.get()=='No Fine-Tuning':
            if Poseactivate == True:
                # pass prompt and image to pipeline with new parameters
                image = pipeline(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=x,
                    height=y,
                    guidance_scale=cfg_value,
                    num_inference_steps=steps_value,
                    controlnetpose=poseimage
                ).images[0]
                image.save(f".//temp//imagegenerate_{t}.png")
                image.save(f".//imagegenerate.png")
                print(f"pose2: {Poseactivate}")
                if (refine.get()== 'Refine Image'):
                    refine_f(f".//imagegenerate.png")
                
                t=t+1
                if t>=10:
                    t=1
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # pass prompt and image to pipeline with new parameters
                image = pipeline(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=x,
                    height=y,
                    guidance_scale=cfg_value,
                    num_inference_steps=steps_value
                ).images[0]
                image.save(f".//temp//imagegenerate_{t}.png")
                image.save(f".//imagegenerate.png")
                if (refine.get()== 'Refine Image'):
                    refine_f(f".//imagegenerate.png")
                t=t+1
                if t>=10:
                    t=1
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
        elif fineturring.get()=='train':
            webbrowser.open('https://colab.research.google.com/github/hollowstrawberry/kohya-colab/blob/main/Lora_Trainer.ipynb?authuser=1#scrollTo=OglZzI_ujZq-')
        else:
            # Percorso del modello personalizzato
            modelfinetur = f".//Fine-Tuning//{fineturring.get()}.safetensors"
            print(f"Modello: {modelfinetur}")

            # Verifica se il pose è attivato
            if Poseactivate:
                print("POSING OK")
                
                # Cattura l'immagine del canvas
                xx, yy = framepose.winfo_rootx(), framepose.winfo_rooty()
                x1, y1 = x + framepose.winfo_width(), y + framepose.winfo_height()
                ImageGrab.grab().crop((xx, yy, x1, y1)).save(".//image_pose.png")
                time.sleep(1)
                
                # Prepara l'immagine del pose
                original_image = Image.open(".//image_pose.png")
                openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
                poseimage = openpose(original_image)
                # Crea e sostituisci lo scheduler DDIM
                scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
                # Carica ControlNet
                controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
                
                if model.get() == 'model_v7':
                    pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16, variant="fp16",use_safetensors=True,scheduler=scheduler) 
                else:
                    pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16,scheduler=scheduler) 
                pipeline.load_lora_weights(modelfinetur, weight_name=f"{fineturring.get()}.safetensors")
                
                # Aggiungi ControlNet alla pipeline
                pipeline.controlnet = controlnet
                
                
                pipeline.scheduler = scheduler
                
                pipeline.to("cuda")
                pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
                
                # Prepara i prompt
                prompt = traduci_ita_eng(text.get("1.0", "end-1c"))
                prompt=prompt+"<lora:marica:1.0>"
                if "a photo of marica" in prompt:
                    pass
                else:
                     prompt= "a photo of marica "+prompt
                negative_prompt = traduci_ita_eng(textneg.get("1.0", "end-1c"))
                
                # Usa Compel per il conditioning
                compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                conditioning = compel(prompt)
                negative_conditioning = compel(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
                
                # Genera l'immagine
                image = pipeline(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=x,
                    height=y,
                    guidance_scale=float(CFG.get()),
                    num_inference_steps=int(STEPS.get()),
                    image=poseimage
                ).images[0]
                
                # Salva l'immagine
                image.save(f".//temp//imagegenerate_{t}.png")
                image.save(f".//imagegenerate.png")
                
                # Refine image se necessario
                if refine.get() == 'Refine Image':
                    refine_f(f".//imagegenerate.png")
                
                # Incrementa il contatore e pulisci la memoria
                t = (t + 1) % 10 if t < 10 else 1
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # Percorso del modello personalizzato
                modelfinetur = f".//Fine-Tuning//{fineturring.get()}.safetensors"
                print(f"Modello: {modelfinetur}")

                # Crea lo scheduler DDIM
                scheduler = DDIMScheduler.from_pretrained(modelid, subfolder="scheduler")

                # Carica il pipeline con lo scheduler personalizzato
                if model.get() == 'model_v7':
                    pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16, variant="fp16",use_safetensors=True,scheduler=scheduler) 
                else:
                    pipeline = AutoPipelineForText2Image.from_pretrained(modelid, torch_dtype=torch.float16,scheduler=scheduler) 
                pipeline.load_lora_weights(modelfinetur, weight_name=f"{fineturring.get()}.safetensors")
                pipeline.to("cuda")
                pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
                
                prompt = traduci_ita_eng(text.get("1.0", "end-1c"))
                prompt=prompt+"<lora:marica:1.0>"
                if "a photo of marica" in prompt:
                    pass
                else:
                     prompt= "a photo of marica "+prompt
                     
                compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
                conditioning = compel(prompt)
                negative_prompt = traduci_ita_eng(textneg.get("1.0", "end-1c"))
                negative_conditioning = compel(negative_prompt)
                [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
                
                image = pipeline(
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=negative_conditioning,
                    width=x,
                    height=y,
                    guidance_scale=float(CFG.get()),
                    num_inference_steps=int(STEPS.get()),
                    
                ).images[0]
                image.save(f".//temp//imagegenerate_{t}.png")
                image.save(f".//imagegenerate.png")
                if (refine.get()== 'Refine Image'):
                    refine_f(f".//imagegenerate.png")
                t=t+1
                if t>=10:
                    t=1
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
    elif funzione.get() == 'Rileva Vestiti':
        if maskesistente== False:
            # Apri l'immagine della maschera
            mask = Image.open("./mask.jpg")

            # Converti la maschera in un array numpy
            mask_array = np.array(mask)

            # Riempi i buchi nella maschera
            filled_mask = ndimage.binary_fill_holes(mask_array > 0)

            # Applica una chiusura morfologica con un elemento strutturante più grande
            smooth_mask = morphology.binary_closing(filled_mask, morphology.disk(10))

            # Applica un filtro gaussiano per smussare ulteriormente i bordi
            smooth_mask = filters.gaussian(smooth_mask, sigma=5)

            # Applica una soglia per riportare l'immagine a binaria
            smooth_mask = smooth_mask > 0.5

            # Converti in un'immagine PIL
            final_mask = Image.fromarray((smooth_mask * 255).astype(np.uint8))

            # Ridimensiona la maschera per farla corrispondere alle dimensioni dell'immagine iniziale
            final_mask = final_mask.resize(init_image.size, Image.LANCZOS)

            # Salva la maschera finale
            final_mask.save('mask_final.png')
        else:
            final_mask= Image.open(".//mask_final.png")
        if Poseactivate==True:
            # Use the final_mask in your pipeline and inpainting
            image = pipeline(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                image=init_image,
                width=x,
                height=y,
                guidance_scale=cfg_value,
                num_inference_steps=steps_value,
                strength=strength_value,
                eta=eta_value,
                mask_image=final_mask,
                controlnet_conditioning_image=poseimage
            ).images[0]
            image.save(f".//temp//imagegenerate_{t}.png")
            image.save(f".//imagegenerate.png")
            if (refine.get()== 'Refine Image'):
                refine_f(f".//imagegenerate.png")
            
            t=t+1
            if t>=10:
                t=1
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
        else:
            # Use the final_mask in your pipeline
           
            image = pipeline(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                image=init_image,
                width=x,
                height=y,
                guidance_scale=cfg_value,
                num_inference_steps=steps_value,
                strength=strength_value,
                eta=eta_value,
                mask_image=final_mask
            ).images[0]
            image.save(f".//temp//imagegenerate_{t}.png")
            image.save(f".//imagegenerate.png")
            if (refine.get()== 'Refine Image'):
                refine_f(f".//imagegenerate.png")
            
            t=t+1
            if t>=10:
                t=1
            del pipeline
            torch.cuda.empty_cache()
            gc.collect()
    elif funzione.get()=='Cambia Sfondo':
        # Use the final_mask in your pipeline
        final_mask= Image.open(".//mask_final.png")
        final_mask= final_mask.resize((init_image.size),Image.LANCZOS)
        image = pipeline(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            image=init_image,
            width=x,
            height=y,
            guidance_scale=cfg_value,
            num_inference_steps=steps_value,
            strength=strength_value,
            eta=eta_value,
            mask_image=final_mask
        ).images[0]
        image.save(f".//temp//imagegenerate_{t}.png")
        image.save(f".//imagegenerate.png")
        if (refine.get()== 'Refine Image'):
            refine_f(f".//imagegenerate.png")
            
        t=t+1
        if t>=10:
           t=1
        
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
    # Calcola le nuove dimensioni mantenendo le proporzioni
    if os.path.exists(".//Imagerefine.png") and refine.get()=='Refine Image':
        image= Image.open(".//Imagerefine.png")
    canvas_ratio = frame1.winfo_width() / frame1.winfo_height()
    img_ratio = image.width / image.height

    if img_ratio > canvas_ratio:
        new_width = frame1.winfo_width()
        new_height = int(new_width / img_ratio)
    else:
        new_height = frame1.winfo_height()
        new_width = int(new_height * img_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(image)

    # Centra l'immagine nel canvas
    x = (frame1.winfo_width() - new_width) // 2
    y = (frame1.winfo_height() - new_height) // 2

    frame1.delete("all")  # Rimuove l'immagine precedente
    frame1.create_image(x, y, anchor='nw', image=img_tk)
    frame1.image = img_tk  # Mantieni un riferimento
    frame1.update()

def esci():
    import sys
    sys.exit()

raccoglitore_points=[]
class PoseEditor:
    def __init__(self, framepose):
        self.canvas = framepose
        
        self.joints = {
            'head': (256, 80),
            'neck': (256, 120),
            'r_shoulder': (226, 120),
            'r_elbow': (196, 180),
            'r_hand': (166, 240),
            'l_shoulder': (286, 120),
            'l_elbow': (316, 180),
            'l_hand': (346, 240),
            'torso': (256, 220),
            'r_hip': (236, 280),
            'r_knee': (236, 380),
            'r_foot': (236, 480),
            'l_hip': (276, 280),
            'l_knee': (276, 380),
            'l_foot': (276, 480)
        }

        self.lines = [
            ('head', 'neck', 'purple'),
            ('neck', 'r_shoulder', 'red'), ('r_shoulder', 'r_elbow', 'orange'), ('r_elbow', 'r_hand', 'yellow'),
            ('neck', 'l_shoulder', 'red'), ('l_shoulder', 'l_elbow', 'orange'), ('l_elbow', 'l_hand', 'yellow'),
            ('neck', 'torso', 'cyan'),
            ('torso', 'r_hip', 'green'), ('r_hip', 'r_knee', 'green'), ('r_knee', 'r_foot', 'lightgreen'),
            ('torso', 'l_hip', 'blue'), ('l_hip', 'l_knee', 'blue'), ('l_knee', 'l_foot', 'lightblue')
        ]

        self.selected_joint = None
        self.draw_skeleton()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def draw_skeleton(self):
        self.canvas.delete("skeleton")
        for start, end, color in self.lines:
            start_pos = self.joints[start]
            end_pos = self.joints[end]
            self.canvas.create_line(start_pos, end_pos, width=8, fill=color, tags="skeleton")

        for joint, pos in self.joints.items():
            self.canvas.create_oval(pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5, fill="white", outline="black", tags="skeleton")

    def on_click(self, event):
        self.selected_joint = min(self.joints, key=lambda j: math.hypot(event.x - self.joints[j][0], event.y - self.joints[j][1]))

    def on_drag(self, event):
        if self.selected_joint:
            self.joints[self.selected_joint] = (event.x, event.y)
            self.draw_skeleton()
def movepose():
    global Poseactivate, framepose, editor
    if Poseactivate == False:
        Poseactivate = True
        editor = PoseEditor(framepose)
    else:
        Poseactivate = False
        framepose.delete("all")
        framepose.configure(bg="black") 
def text_button():
    global BUTTONGEN, funzione
    if funzione.get() == 'ImageToImage':
        BUTTONGEN.config(text='Genera ImgToImg')
    elif funzione.get() == 'Rileva Vestiti':
        BUTTONGEN.config(text='Rileva Vestiti')
    elif funzione.get() == 'Genera Immagini':
        BUTTONGEN.config(text='Genera Immagini')
    elif funzione.get()=='Video': 
        BUTTONGEN.config(text='VIDEO')   
    elif funzione.get()=='Estrai Fotogrammi':
        BUTTONGEN.config(text='Estrai Fotogrammi') 

# Il resto del codice rimane invariato
window = TkinterDnD.Tk()
window.title("IMAGE TO TRASFORM(INPUT IMAGE)")
window.geometry('1000x1000')
window.attributes('-fullscreen', True)

# Frame per contenere le canvas di input
frame_frames_input = Frame(window)
frame_frames_input.grid(row=0, column=0, padx=0, sticky="nw")

frameface = Canvas(frame_frames_input, width=256, height=256, bg='pink')
frameface.grid(row=0, column=0, padx=1,sticky="nw")
frameface.drop_target_register(DND_FILES)
frameface.dnd_bind('<<Drop>>', lambda event: drop_image_face(event, frameface))

frameimg_input = Canvas(frame_frames_input, width=256, height=256, bg='pink')
frameimg_input.grid(row=1, column=0, padx=0,sticky="nw")
frameimg_input.drop_target_register(DND_FILES)
frameimg_input.dnd_bind('<<Drop>>', lambda event: drop_image(event, frameimg_input))

framecanvas = Frame(window)
framecanvas.grid(row=0, column=0, padx=300, sticky="nw")

# Frame principale affiancato alle canvas di input
frame1 = Canvas(framecanvas, width=960, height=720, bg='red')
frame1.grid(row=0, column=1, padx=0, pady=1, sticky="nw")
frame1.drop_target_register(DND_FILES)
frame1.dnd_bind('<<Drop>>', lambda event: drop_image(event, frame1))

# Canvas framepose
framepose = Canvas(framecanvas, width=512, height=512, bg='black')
framepose.grid(row=0, column=2, padx=0, pady=0, sticky="nw")

# Bottoni
framebottoneframe = Frame(framecanvas)  # Cambio: ora è un figlio di framecanvas
framebottoneframe.grid(row=1, column=2, padx=0, sticky="nw")  # Cambio: ora è sotto framepose

labedichette= Label(framebottoneframe,text="Funzioni")
labedichette.grid(row=0,column=1)
labedichette2= Label(framebottoneframe,text="Fine Tuning")
labedichette2.grid(row=0,column=2)

pose_button = Button(framebottoneframe, text="Attiva Editor Pose", command=movepose)
pose_button.grid(row=1, column=0, padx=0, pady=0, sticky="nw")

funzione = Combobox(framebottoneframe, values=['ImageToImage', 'Rileva Vestiti', 'Genera Immagini', 'Cambia Sfondo','Estrai Fotogrammi','Video'])
funzione.set('ImageToImage')
funzione.grid(row=1, column=1, padx=0, pady=0, sticky="nw")
funzione.bind('<<ComboboxSelected>>', lambda event: text_button())

def modelfineF():
    global fineturring
    modelfine = [os.path.splitext(m)[0] for m in os.listdir('./Fine-Tuning') if m.endswith(('.ckpt', '.safetensors'))]
    fineturring.config(values=['No Fine-Tuning','train'] + modelfine)

fineturring = Combobox(framebottoneframe, values=['No Fine-Tuning','train'])
fineturring.grid(row=1, column=2, padx=0, pady=0, sticky="nw")
fineturring.set('No Fine-Tuning')
modelfineF()
# Funzione per aggiornare il prompt negativo
def changeneg(event=None):
    global fineturring
    promptnegativefineturing = """(intimo),(costume da bagno),(reggiseno),(slip),(vestiti),(gruppo, persone multiple, soggetti multipli, persona extra, due persone,duplicato),deformato,distorto,
sfigurato,mal disegnato,anatomia errata,arto extra,arto mancante,arti fluttuanti,arti sconnessi,mutazione,mutato,brutto,disgustoso,sfocato,fuori fuoco,noioso,arte scadente,cartone animato,anime,render 3d,
CGI,illustrazione,dipinto,schizzi"""

    if fineturring.get() != 'train' and fineturring.get() != 'No Fine-Tuning':
        textneg.delete('1.0', 'end')  # Cancella il contenuto esistente
        textneg.insert('1.0', promptnegativefineturing)  # Inserisci il nuovo contenuto
# Associa la funzione changeneg all'evento di selezione della Combobox
fineturring.bind('<<ComboboxSelected>>', changeneg)


class PenTool:
    def __init__(self, canvas):
        global maskesistente,BUTTONGEN
        self.maskesistente = maskesistente
        self.canvas = canvas
        self.points = []
        self.lines = []
        self.handles = []
        self.active_point = None
        self.resized_image_size = (0, 0)  # Inizializza resized_image_size
        self.init_image_size = (0, 0)  # Inizializza init_image_size
        self.canvas.bind('<Button-1>', self.add_or_select_point)
        self.canvas.bind('<B1-Motion>', self.move_point)
        self.canvas.bind('<ButtonRelease-1>', self.release_point)
        self.canvas.bind('<Motion>', self.highlight_point)
        
        # aggiungi resetta points premi rotellina mouse
        self.canvas.bind('<Button-2>', self.resetpoint)
        
        # aggiungi elimina mask
        self.canvas.bind('<Button-3>', self.delete_mask)
    
    def delete_mask(self, event):
        # Rimuovi i file delle maschere
        if os.path.exists(".//mask_final.png"):
            os.remove(".//mask_final.png")
        if os.path.exists(".//mask_penna.png"):
            os.remove(".//mask_penna.png")
        
        # Rimuovi gli elementi grafici relativi ai punti e alle linee
        for handle in self.handles:
            self.canvas.delete(handle)
        for line in self.lines:
            self.canvas.delete(line)
        
        # Resetta le liste
        self.points = []
        self.lines = []
        self.handles = []
        
        # Resetta la variabile globale maskesistente
        global maskesistente
        maskesistente = False
         
    def resetpoint(self, event):
        # Resetta le liste
        self.points = []
        self.lines = []
        self.handles = []

    def add_or_select_point(self, event):
        x, y = event.x, event.y
        for i, (px, py) in enumerate(self.points):
            if (px-4 <= x <= px+4) and (py-4 <= y <= py+4):
                self.active_point = i
                return
        self.add_point(x, y)

    def add_point(self, x, y):
        global BUTTONGEN
        BUTTONGEN.config(text="Inpainting")
        self.points.append((x, y))
        handle = self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='red', tags='handle')
        self.handles.append(handle)
        if len(self.points) > 1:
            line = self.canvas.create_line(self.points[-2], self.points[-1], fill='white')
            self.lines.append(line)
        self.update_mask()

    def move_point(self, event):
        if self.active_point is not None:
            x, y = event.x, event.y
            self.points[self.active_point] = (x, y)
            self.canvas.coords(self.handles[self.active_point], x-4, y-4, x+4, y+4)
            if self.active_point > 0:
                self.canvas.coords(self.lines[self.active_point-1], 
                                   self.points[self.active_point-1][0],
                                   self.points[self.active_point-1][1],
                                   x, y)
            if self.active_point < len(self.points) - 1:
                self.canvas.coords(self.lines[self.active_point], 
                                   x, y,
                                   self.points[self.active_point+1][0],
                                   self.points[self.active_point+1][1])
            self.update_mask()

    def release_point(self, event):
        self.active_point = None

    def highlight_point(self, event):
        x, y = event.x, event.y
        for i, (px, py) in enumerate(self.points):
            if (px-4 <= x <= px+4) and (py-4 <= y <= py+4):
                self.canvas.itemconfig(self.handles[i], fill='blue')
            else:
                self.canvas.itemconfig(self.handles[i], fill='red')

    def update_mask(self):
        global maskesistente
        if len(self.points) > 2:
        # Open the initial image
            init_image = Image.open(".//temp_image.png")
            
            # Calculate the new dimensions while maintaining the aspect ratio
            canvas_width, canvas_height = 960, 720
            img_ratio = init_image.width / init_image.height
            
            if img_ratio > canvas_width / canvas_height:
                new_width = canvas_width
                new_height = int(new_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(new_height * img_ratio)
            
            # Center the image in the canvas
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Create a black image with the dimensions of the canvas
            mask = Image.new('L', (canvas_width, canvas_height), 0)
            draw = ImageDraw.Draw(mask)
            
            # Draw the polygon on the mask using the original points
            draw.polygon(self.points, fill=255)
            
            # Crop the mask to the new dimensions, ensuring symmetric cropping
            left = x_offset
            upper = y_offset
            right = x_offset + new_width
            lower = y_offset + new_height
            
            mask_cropped = mask.crop((left, upper, right, lower))
            
            maskresize = mask_cropped.resize(init_image.size, Image.LANCZOS)
            
            # Ensure dimensions are always divisible by 8
            x, y = maskresize.size
            
            xd = (x // 8) * 8
            yd = (y // 8) * 8

            # Resize the input image if necessary
            if maskresize.size != (xd, yd):
                maskresize = maskresize.resize((xd, yd), Image.LANCZOS)
            
            if os.path.exists(".//mask_final.png"):
                mask_final = Image.open(".//mask_final.png")
                # Combine the masks without overwriting existing white areas
                combined_mask = ImageChops.lighter(mask_final, maskresize)
                combined_mask.save("mask_final.png")
            else:
                maskresize.save("mask_final.png")
            
            # Save the final mask
            maskresize.save("mask_penna.png")
            maskesistente= True


pen_tool = PenTool(frame1)

# Label e Text sotto frame1 
frame_text = Frame(window) 
frame_text.grid(row=1, column=0,padx=300, pady=1,sticky='sw')  
l1 = Label(frame_text, text="prompt") 
l1.grid(row=0, column=0) 
text = Text(frame_text, width=100, height=5) 
text.grid(row=1, column=0)  
l2 = Label(frame_text, text="prompt Negativo") 
l2.grid(row=2, column=0) 
textneg = Text(frame_text, width=100, height=5) 
textneg.grid(row=3, column=0),
# Frame di controllo sotto le Text 
framecontrol = Frame(window) 
framecontrol.grid(row=3, column=0, padx=0,pady=10,sticky='sw')


negativedefine="""multiple people, crowd, group, many subjects, extra characters, additional figures, High pass filter, 
airbrush, portrait, zoomed, soft light, deformed, extra limbs, extra fingers, mutated hands, bad anatomy, bad proportions, 
blind, bad eyes, ugly eyes, dead eyes, blur, vignette, out of shot, out of focus, monochrome, grainy, noisy, text, writing, watermark, 
logo, oversaturation, over saturation, over shadow, out of frame, obese, odd proportions, asymmetrical, fat, dialog, words, fonts, teeth, ((((ugly)))), 
(((duplicate))), ((morbid)), b&w, [out of frame], ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), blurry, cloned face, (((disfigured))), gross proportions, 
(malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), neon lights, tan lines, wrinkles, armpit hair, 
lowres, poorly drawn, crippled, crooked, broken, weird, odd, distorted, erased, cut, mutilated, sloppy, hideous, pixelated, aliasing, mutated, extra_limb, two heads, 
children, gross, disgusting, horrible, scary, evil, old, conjoined, morphed, error, glitch, extra digits, signature, jpeg artifacts, low quality, unfinished, cropped, siamese twins, robot eyes, 
asian, penis, cartoon, manga"""
textneg.insert("1.0", negativedefine)




def cambia_prompt_negativo(event=None):
    global negativedefine, negativedefine2, textneg, model
    
    # Pulisci il contenuto attuale del textneg
    textneg.delete("1.0", END)
    
    if model.get() == 'model_v7':
        negativedefine2 = """multiple people, crowd, group, many subjects, extra characters, additional figures, High pass filter, 
airbrush, portrait, zoomed, soft light, deformed, extra limbs, extra fingers, mutated hands, bad anatomy, bad proportions, 
blind, bad eyes, ugly eyes, dead eyes, blur, vignette, out of shot, out of focus, monochrome, grainy, noisy, text, writing, watermark, 
logo, oversaturation, over saturation, over shadow, out of frame, obese, odd proportions, asymmetrical, fat, dialog, words, fonts, teeth, ((((ugly)))), 
(((duplicate))), ((morbid)), b&w, [out of frame], ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), blurry, cloned face, (((disfigured))), gross proportions, 
(malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), neon lights, tan lines, wrinkles, armpit hair, 
lowres, poorly drawn, crippled, crooked, broken, weird, odd, distorted, erased, cut, mutilated, sloppy, hideous, pixelated, aliasing, mutated, extra_limb, two heads, , gross, disgusting, horrible, scary, evil, old, conjoined, morphed, error, glitch, extra digits, signature, jpeg artifacts, low quality, unfinished, cropped, siamese twins, robot eyes, 
asian, penis, cartoon, manga"""
        textneg.insert("1.0", negativedefine2)
    else:
        negativedefine = """multiple people, crowd, group, many subjects, extra characters, additional figures, High pass filter, 
airbrush, portrait, zoomed, soft light, deformed, extra limbs, extra fingers, mutated hands, bad anatomy, bad proportions, 
blind, bad eyes, ugly eyes, dead eyes, blur, vignette, out of shot, out of focus, monochrome, grainy, noisy, text, writing, watermark, 
logo, oversaturation, over saturation, over shadow, out of frame, obese, odd proportions, asymmetrical, fat, dialog, words, fonts, teeth, ((((ugly)))), 
(((duplicate))), ((morbid)), b&w, [out of frame], ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), blurry, cloned face, (((disfigured))), gross proportions, 
(malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), neon lights, tan lines, wrinkles, armpit hair, 
lowres, poorly drawn, crippled, crooked, broken, weird, odd, distorted, erased, cut, mutilated, sloppy, hideous, pixelated, aliasing, mutated, extra_limb, two heads, , gross, disgusting, horrible, scary, evil, old, conjoined, morphed, error, glitch, extra digits, signature, jpeg artifacts, low quality, unfinished, cropped, siamese twins, robot eyes, 
asian, penis, cartoon, manga"""
        textneg.insert("1.0", negativedefine)
    
    

    

# Modello
labmodel = Label(framecontrol, text="Seleziona Modello")
labmodel.grid(row=0, column=0, pady=(5,0), padx=5, sticky='sw')

model = Combobox(framecontrol, values=['model_v7', 'realvisxlv60-B1','Rocco_v1', 'Rocco_v2'])
model.set('model_v7')
model.grid(row=1, column=0, pady=(0,5), padx=5, sticky='nw')
model.bind("<<ComboboxSelected>>", cambia_prompt_negativo)


# Chiamata iniziale per impostare il prompt negativo corretto all'avvio
cambia_prompt_negativo()



# Risoluzione
labrisol = Label(framecontrol, text="Risoluzione")
labrisol.grid(row=0, column=1, pady=(5,0), padx=5, sticky='sw')

risoluzione = Combobox(framecontrol, values=['image Input', '512,720', '720,512', '960,720', '720,960'])
risoluzione.set('image Input')
risoluzione.grid(row=1, column=1, pady=(0,5), padx=5, sticky='nw')

BUTTONGEN = Button(framecontrol, text='Genera ImgToImg', command=imgtoimag)
BUTTONGEN.grid(row=1, column=2, pady=(0,5), padx=5, sticky='nw')

DPM= Combobox(framecontrol,values=['None','DPM'])
DPM.grid(row=1, column=3, pady=(0,5), padx=5, sticky='nw')

interoga = Button(framecontrol, text="Interoga", foreground='blue', command=clip_interogation)
interoga.grid(row=1, column=4, pady=(0,5), padx=5, sticky='nw')

def deepFace():
    global frame1, image
    pathgen = None
    if os.path.exists(".//Imagerefine.png"):
        pathgen = ".//Imagerefine.png"
    elif os.path.exists(".//imagegenerate.png"):
        pathgen = ".//imagegenerate.png"
        
    if pathgen is not None and os.path.exists(".//temp_image.png"):
        if os.path.exists(".//FaceDancer//temp_face.png"):
            os.remove(".//FaceDancer//temp_face.png")
        if os.path.exists(".//temp_face.png"):
            shutil.copyfile(".//temp_face.png", ".//FaceDancer//temp_face.png")
        else:
            shutil.copyfile(".//temp_image.png", ".//FaceDancer//temp_face.png")
            
        if os.path.exists(".//FaceDancer//imagegenerate.png"):
            os.remove(".//FaceDancer//imagegenerate.png")
        shutil.move(pathgen, ".//FaceDancer//imagegenerate.png")
        os.chdir("FaceDancer")
        os.system('python test_image_swap_multi.py --facedancer_path "./model_zoo/FaceDancer_config_c_HQ.h5" --img_path ".//imagegenerate.png" --swap_source ".//temp_face.png" --img_output "results/swapped_image.png"')
        os.chdir("..")
        if os.path.exists(".//FaceDancer//results/swapped_image.png"):
            shutil.copyfile(".//FaceDancer//results/swapped_image.png", ".//imagegenerate.png")
            image = Image.open(".//imagegenerate.png")
            # Calcola le nuove dimensioni mantenendo le proporzioni
            canvas_ratio = frame1.winfo_width() / frame1.winfo_height()
            img_ratio = image.width / image.height

            if img_ratio > canvas_ratio:
                new_width = frame1.winfo_width()
                new_height = int(new_width / img_ratio)
            else:
                new_height = frame1.winfo_height()
                new_width = int(new_height * img_ratio)

            image = image.resize((new_width, new_height), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image)

            # Centra l'immagine nel canvas
            x = (frame1.winfo_width() - new_width) // 2
            y = (frame1.winfo_height() - new_height) // 2

            frame1.delete("all")  # Rimuove l'immagine precedente
            frame1.create_image(x, y, anchor='nw', image=img_tk)
            frame1.image = img_tk  # Mantieni un riferimento
            frame1.update()

deepface = Button(framecontrol, text="cambia viso", foreground='orange',command=deepFace)
deepface.grid(row=1, column=5, pady=(0,5), padx=5, sticky='nw')


BUTTONexit = Button(framecontrol, text="Exit", foreground='red', command=esci)
BUTTONexit.grid(row=1, column=6, pady=(0,5), padx=5, sticky='nw')

def downCFG():
    global CFG
    CFG.set(max(CFG.get() - 0.1, 1.0))

def upCFG():
    global CFG
    CFG.set(min(CFG.get() + 0.1, 50.0))

def downSTEPS():
    global STEPS
    STEPS.set(max(STEPS.get() - 1, 1))

def upSTEPS():
    global STEPS
    STEPS.set(min(STEPS.get() + 1, 150))

def downStrength():
    global strength
    strength.set(max(strength.get() - 0.01, 0.1))

def upStrength():
    global strength
    strength.set(min(strength.get() + 0.01, 1.0))

def downEta():
    global eta
    eta.set(max(eta.get() - 0.01, 0.1))

def upEta():
    global eta
    eta.set(min(eta.get() + 0.01, 1.0))

buttonDownCFG = Button(framecontrol, text='<-', command=downCFG)
buttonDownCFG.grid(row=1, column=7, pady=(0,5), padx=5, sticky='nw')
buttonUPCFG = Button(framecontrol, text='->', command=upCFG)
buttonUPCFG.grid(row=1, column=8, pady=(0,5), padx=5, sticky='nw')

CFG = Scale(framecontrol, from_=1.0, to=50.0, orient=HORIZONTAL, label="CFG", resolution=0.1)
CFG.grid(row=1, column=9, pady=(0,5), padx=5, sticky='nw')
CFG.set(1.5)

buttonDownSTEPS = Button(framecontrol, text='<-', command=downSTEPS)
buttonDownSTEPS.grid(row=1, column=10, pady=(0,5), padx=5, sticky='nw')
buttonUPSTEPS = Button(framecontrol, text='->', command=upSTEPS)
buttonUPSTEPS.grid(row=1, column=11, pady=(0,5), padx=5, sticky='nw')

STEPS = Scale(framecontrol, from_=1, to=150, orient=HORIZONTAL, label="Steps")
STEPS.grid(row=1, column=12, pady=(0,5), padx=5, sticky='nw')
STEPS.set(5)

buttonDownStrength = Button(framecontrol, text='<-', command=downStrength)
buttonDownStrength.grid(row=1, column=13, pady=(0,5), padx=5, sticky='nw')
buttonUPStrength = Button(framecontrol, text='->', command=upStrength)
buttonUPStrength.grid(row=1, column=14, pady=(0,5), padx=5, sticky='nw')

strength = Scale(framecontrol, from_=0.1, to=1.0, orient=HORIZONTAL, label="Strength", resolution=0.01)
strength.grid(row=1, column=15, pady=(0,5), padx=5, sticky='nw')
strength.set(0.92)

buttonDownEta = Button(framecontrol, text='<-', command=downEta)
buttonDownEta.grid(row=1, column=16, pady=(0,5), padx=5, sticky='nw')
buttonUPEta = Button(framecontrol, text='->', command=upEta)
buttonUPEta.grid(row=1, column=17, pady=(0,5), padx=5, sticky='nw')

eta = Scale(framecontrol, from_=0.1, to=1.0, orient=HORIZONTAL, label="Eta", resolution=0.01)
eta.grid(row=1, column=18, pady=(0,5), padx=5, sticky='nw')
eta.set(1.0)


refine= Combobox(framecontrol,values=['Refine Image','no Refine Image'])
refine.grid(row=1,column=19,pady=(0,5), padx=5,sticky='nw')

def saveImage():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        if os.path.exists(".//imagegenerate.png"):
            shutil.copyfile(".//imagegenerate.png", os.path.join(os.path.dirname(file_path), os.path.basename(file_path)))

            
save= Button(framecontrol,text="Salva Image",command= saveImage)
save.grid(row=1,column=20, pady=(0,5), padx=5,sticky='nw')

def cercafoto_F():
    url = 'https://www.reverseimagesearch.com/'
    webbrowser.open(url)
    
google_cerca= Button(framecontrol,text="Cerca foto",command=cercafoto_F)
google_cerca.grid(row=1,column=21,pady=(0,5),padx=5,sticky='nw')

window.mainloop()