import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from skimage import color
import numpy as np
import torchvision.transforms as transforms
from model2color import NeuralNet  # Zaimportuj swój model
import warnings
from skimage import color

class ColorPredictorApp:
    def __init__(self, master, model_path):
        self.master = master
        self.model_path = model_path
        self.current_image = None
        self.model = None
        self.setup_model()
        self.setup_gui()

    def setup_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup_gui(self):
        self.master.title("Color Predictor")
        self.frame = ttk.Frame(self.master)
        self.frame.pack(padx=10, pady=10)

        # Przyciski
        self.btn_frame = ttk.Frame(self.frame)
        self.btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)

        # Canvas dla obrazu
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()

        # Wyświetlanie kolorów
        self.color_frame = ttk.Frame(self.frame)
        self.color_frame.pack(pady=10)
        
        self.color1_label = ttk.Label(self.color_frame, width=20, relief='solid')
        self.color1_label.pack(side=tk.LEFT, padx=10, ipady=30, ipadx=30)
        
        self.color2_label = ttk.Label(self.color_frame, width=20, relief='solid')
        self.color2_label.pack(side=tk.LEFT, padx=10, ipady=30, ipadx=30)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.current_image = Image.open(path).convert('RGB')
            self.display_image()

    def display_image(self):
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.axis('off')
        self.canvas.draw()



    def lab_to_rgb(self, lab_normalized):
        # Denormalizacja LAB
        lab_denormalized = [
            lab_normalized[0] * 100,          # L: 0-100
            (lab_normalized[1] * 255) - 128,   # a: -128-127
            (lab_normalized[2] * 255) - 128    # b: -128-127
        ]
        
        # Przycięcie wartości LAB do dopuszczalnego zakresu
        lab_clipped = np.clip(lab_denormalized, [0, -128, -128], [100, 127, 127])
        
        # Konwersja do RGB
        lab_array = np.array([lab_clipped], dtype=np.float64)  # Kształt (1, 3)
        rgb = color.lab2rgb(lab_array)[0]  # Zwraca kształt (3,)
        
        # Przycięcie i konwersja na wartości 0-255
        rgb_clipped = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb_clipped

    def predict(self):
        if self.current_image is None:
            return

        # Przetwarzanie obrazu
        input_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()[0]

        # Podział na dwa kolory
        color1 = output[:3]
        color2 = output[3:]
        
        # Konwersja do RGB
        rgb1 = self.lab_to_rgb(color1)
        rgb2 = self.lab_to_rgb(color2)

        # Aktualizacja GUI
        self.color1_label.config(background=f'#{rgb1[0]:02x}{rgb1[1]:02x}{rgb1[2]:02x}')
        self.color2_label.config(background=f'#{rgb2[0]:02x}{rgb2[1]:02x}{rgb2[2]:02x}')

if __name__ == '__main__':
    root = tk.Tk()
    app = ColorPredictorApp(root, 'color_predictor_lab_two_colors.pth')
    root.mainloop()