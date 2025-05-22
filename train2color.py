import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset2color import ColorDataset
from model2color import NeuralNet

# Konfiguracja
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 60
MODEL_PATH = 'color_predictor_lab_two_colors.pth'

def main():
    # Dataset i DataLoader

    print(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = ColorDataset(
        images_dir='C:/Users/agnel/Desktop/BIAI dataset/photos/PhotosColorPicker',
        labels_dir='C:/Users/agnel/Desktop/BIAI dataset/user data/klasyfikacja/',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, funkcja straty, optymalizator
    model = NeuralNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Trening
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:  # Tylko dwa elementy!
            inputs = inputs.to(device)
            targets = targets.to(device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            outputs = model(inputs)  # Kształt: (batch_size, 6)
            
            # Podział wyjść na dwa kolory
            color_primary_pred = outputs[:, :3]
            color_secondary_pred = outputs[:, 3:]
            
            # Podział celów na dwa kolory
            color_primary_true = targets[:, :3]
            color_secondary_true = targets[:, 3:]
            
            # Oblicz stratę dla obu kolorów
            loss_primary = criterion(color_primary_pred, color_primary_true)
            loss_secondary = criterion(color_secondary_pred, color_secondary_true)
            total_loss = loss_primary + loss_secondary
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
    
        print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader):.4f}')

    # Zapis modelu
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main()