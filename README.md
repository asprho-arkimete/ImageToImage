# ImageToImage
crea ImagetoImage, inpainting, generazione immagine, genera video, estrai forogrammi, addestra modelli lora;
istruzioni 
Clona il repository principale:
git clone https://github.com/asprho-arkimete/ImageToImage.git
cd imagetoimage

Crea un ambiente virtuale:
python3.10 -m venv vimagetoimage
cd vimagetoimage/Scripts
activate

Installa le dipendenze:
pip install -r requirements.txt

Installa CUDA:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Clona e installa i progetti aggiuntivi:
PIA:
cd ../imagetoimage
git clone https://github.com/open-mmlab/PIA.git
cd PIA
pip install -r requirements.txt

FaceDancer:
cd ../imagetoimage
git clone https://github.com/felixrosberg/FaceDancer.git
cd FaceDancer
pip install -r requirements.txt

Crea la cartella per i modelli:
cd ../imagetoimage
mkdir model_v7_inpaint

Scarica i modelli nella cartella “model_v7_inpaint”:
AnalogMadness-realistic-model-v7:
cd model_v7_inpaint
wget digiplay/AnalogMadness-realistic-model-v7

UberRealisticonMergeUrpmV13Inpainting:
wget goldpulpy/UberRealisticPornMergeUrpmV13Inpainting

Altri modelli:
Gli altri modelli saranno scaricati automaticamente. Se ci sono problemi, i link sono forniti nello script.

