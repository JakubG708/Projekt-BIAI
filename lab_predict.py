import torch
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import torchvision.transforms as transforms
from model import NeuralNet
from tkinter import Tk, filedialog

def predict_and_display(image_path=None, model_path='color_predictor_lab.pth'):
    # Inicjalizacja Tkinter
    root = Tk()
    root.withdraw()  # Ukryj główne okno
    
    # Wybór pliku jeśli nie podano ścieżki
    if not image_path:
        image_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Pliki obrazów", "*.jpg *.jpeg *.png *.bmp")]
        )
    
    if not image_path:
        print("Nie wybrano pliku")
        return

    # Konfiguracja
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Wczytaj model
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transformacje
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Wczytaj i przetwórz obraz
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Błąd wczytywania obrazu: {e}")
        return

    # Predykcja
    with torch.no_grad():
        lab_output = model(input_tensor).squeeze().cpu().numpy()

    # Konwersja LAB do RGB
    def denormalize_lab(lab):
        return [
            lab[0] * 100, 
            (lab[1] * 255) - 128, 
            (lab[2] * 255) - 128
        ]
    
    try:
        lab_denormalized = denormalize_lab(lab_output)
        rgb_pred = color.lab2rgb(np.array([lab_denormalized], dtype=np.float64))[0]
        rgb_pred = (rgb_pred * 255).astype(np.uint8)
    except Exception as e:
        print(f"Błąd konwersji koloru: {e}")
        return

    # Wizualizacja
    plt.figure(figsize=(10, 5))
    
    # Oryginalny obraz
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Wybrany obraz')
    plt.axis('off')

    # Przewidziany kolor
    plt.subplot(1, 2, 2)
    color_block = np.zeros((100, 100, 3), dtype=np.uint8)
    color_block[:, :] = rgb_pred
    plt.imshow(color_block)
    plt.title(f'Przewidziany kolor\nRGB: {rgb_pred}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    predict_and_display()