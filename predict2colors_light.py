import torch
from PIL import Image
import torchvision.transforms as transforms
from tkinter import Tk, filedialog
from model2color import NeuralNet  # Zaimportuj swój model

def main():
    # Inicjalizacja modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load('color_predictor_lab_two_colors.pth', map_location=device))
    model.eval()

    # Transformacje
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Wybór obrazu
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if not image_path:
        return

    # Przetwarzanie obrazu
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]

    # Denormalizacja LAB
    def denormalize_lab(lab_norm):
        return [
            lab_norm[0] * 100,          # L
            (lab_norm[1] * 255) - 128,  # a
            (lab_norm[2] * 255) - 128   # b
        ]

    # Podział na dwa kolory
    color1 = denormalize_lab(output[:3])
    color2 = denormalize_lab(output[3:])

    print("Przewidziane wartości LAB:")
    print(f"Kolor 1: L={color1[0]:.1f}, a={color1[1]:.1f}, b={color1[2]:.1f}")
    print(f"Kolor 2: L={color2[0]:.1f}, a={color2[1]:.1f}, b={color2[2]:.1f}")

if __name__ == '__main__':
    main()