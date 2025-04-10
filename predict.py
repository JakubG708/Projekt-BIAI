import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dataset import ColorDataset
from model import NeuralNet

# Konfiguracja
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'color_predictor.pth'

def plot_predictions(model, dataset, num_samples=5):
    model.eval()
    indices = torch.randint(0, len(dataset), (num_samples,))
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))  # 3 kolumny
    
    for i, idx in enumerate(indices):
        image, true_color = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_color = model(image_tensor).squeeze().cpu().numpy()
        
        # Konwersja kolorów do zakresu 0-1
        true_color_rgb = true_color.numpy()
        pred_color_rgb = pred_color

        # Wyświetl obraz
        axs[i, 0].imshow(image.permute(1, 2, 0))
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')
        
        # Wyświetl prawdziwy kolor
        axs[i, 1].imshow([[true_color_rgb]], aspect='auto')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('True Color')
        
        # Wyświetl przewidywany kolor
        axs[i, 2].imshow([[pred_color_rgb]], aspect='auto')
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Predicted Color')
    
    plt.tight_layout()
    plt.show()

def main():
    # Wczytaj model
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = ColorDataset(
        images_dir='C:/Users/agnel/Desktop/BIAI dataset/photos/PhotosColorPicker',
        labels_dir='C:/Users/agnel/Desktop/BIAI dataset/user data/klasyfikacja/',
        transform=transform
    )
    
    # Wyświetl predykcje
    plot_predictions(model, dataset)

if __name__ == '__main__':
    main()