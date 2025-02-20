import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage as sc
from PIL import Image
import cv2  # Importă OpenCV pentru procesarea imaginii

# Transformarea unei imagini cu niveluri aparent de gri într-o imagine reală grayscale
def rgb2gri(img_in, format):
    img_in = img_in.astype('float')
    s = img_in.shape
    if len(s) == 3 and s[2] == 3:  # Verifică dacă imaginea este color
        if format == 'png':
            img_out = (0.299 * img_in[:, :, 0] + 0.587 * img_in[:, :, 1] + 0.114 * img_in[:, :, 2]) * 255
        elif format == 'jpg':
            img_out = 0.299 * img_in[:, :, 0] + 0.587 * img_in[:, :, 1] + 0.114 * img_in[:, :, 2]
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out.astype('uint8')
        return img_out
    else:
        return img_in

# Funcția putere pentru transformare neliniară
def power(img_in, L, r):
    img_out = img_in.astype(float)  # Conversie la float pentru calcul
    img_out = (L - 1) * ((img_out / (L - 1)) ** r)
    img_out = np.clip(img_out, 0, 255)  # Limitare valori între 0 și 255
    img_out = img_out.astype('uint8')  # Conversie la uint8 pentru imagine
    return img_out

# Ajustarea contrastului liniar pe porțiuni
def contrast_liniar_portiuni(img_in, L, a, b, Ta, Tb):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_in[i, j] < a):
                img_out[i, j] = (Ta / a) * img_in[i, j]
            if (img_in[i, j] >= a and img_in[i, j] <= b):
                img_out[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_in[i, j] - a)
            if (img_in[i, j] > b):
                img_out[i, j] = Tb + ((L - 1 - Tb) / (L - 1 - b)) * (img_in[i, j] - b)
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

# Clipping (modificarea liniară a contrastului)
def clipping(img_in, a, b, Ta, Tb):
    """Aplică clipping pe intervalul specificat."""
    img_out = np.empty_like(img_in)
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            if img_in[i, j] <= a:
                img_out[i, j] = 0  # Setează la 0 dacă este sub pragul inferior
            elif a < img_in[i, j] <= b:
                img_out[i, j] = Ta + (Tb - Ta) / (b - a) * (img_in[i, j] - a)
            else:
                img_out[i, j] = 255  # Setează la valoarea maximă dacă este peste pragul superior
    return img_out.astype("uint8")

# Metoda folosită pentru segmentare (binarizare)
def binarizare(img_in, L, a):
    s = img_in.shape
    img_out = np.empty_like(img_in)
    img_in = img_in.astype(float)
    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if (img_in[i, j] < a):
                img_out[i, j] = 0
            else:
                img_out[i, j] = 255
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')
    return img_out

# Funcție pentru calcularea caracteristicilor geometrice (raza, circumferința și aria)
def extragere_caract(img_bin):
 
    # Găsește contururile în imagine
    img_bin = np.uint8(img_bin)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inițializare variabile pentru rază, circumferință și arie:
    radius = 0
    circumference = 0
    area = 0

    for contour in contours:
        # 1. Calcularea ariei folosind funcția OpenCV
        area = cv2.contourArea(contour)
        
        # 2. Calcularea circumferinței folosind perimetrul conturului
        circumference = cv2.arcLength(contour, True)
        
        # 3. Calcularea razei (raza cercului care înconjoară conturul)
        # Folosim cercul minim care înconjoară obiectul
        (x, y), radius_min = cv2.minEnclosingCircle(contour)
        radius = max(radius, radius_min)

    return radius, circumference, area

# Setează calea către folderul cu imagini
folder_path = r"C:/Users/ANA/Desktop/DataSet_IM/imagini"

if not os.path.exists(folder_path):
    print(f"Folderul {folder_path} nu există. Verifică calea!")
else:
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"Nu s-au găsit imagini în folderul: {folder_path}")
    else:
        num_images = len(image_files)
        cols = 3
        rows = (num_images // cols) + (num_images % cols > 0)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        w3 = np.array([[1], [1], [1]])  # Structura pentru binary_closing

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = np.array(Image.open(image_path).convert("L"))
                img_gri = rgb2gri(img, 'jpg')

                # Aplică funcția power
                img_power = power(img_gri, L=255, r=2.5)

                # Aplică funcția contrast pe porțiuni
                img_contrast = contrast_liniar_portiuni(img_gri, L=255, a=100, b=200, Ta=40, Tb=220)

                # Aplică funcția clipping
                img_clipped = clipping(img_gri, a=120, b=200, Ta=0, Tb=220)

                # Aplică funcția binarizare
                img_binary = binarizare(img_gri, L=255, a=175)

                # Aplică operația binary closing
                img_closing = sc.binary_closing(img_binary, structure=w3)

                # Calculul caracteristicilor tumorii folosind noua metodă
                radius, circumference, area = extragere_caract(img_binary)

                print(f"Imagine: {image_file}, Rază: {radius}, Circumferință: {circumference}, Arie: {area}")

                # Afișare imagini prelucrate (comentare/decomentare pentru teste)
                axes[i].imshow(img_binary, cmap="gray")  # Imagine binarizată
                axes[i].set_title(f"{image_file}\nArie: {area},\nCirc: {circumference:.1f},\nRază: {radius:.1f}")
                axes[i].axis("on")
            except Exception as e:
                print(f"Eroare la citirea imaginii {image_file}: {e}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
