import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset2color import ColorDataset
from model2color import NeuralNet

# Konfiguracja
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = 'color_predictor_lab_two_colors.pth'

def main():
    # Dataset i DataLoader

    print(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
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
        
        for inputs, color_primary, color_secondary in dataloader:
            inputs = inputs.to(device)
            color_primary = color_primary.to(device)
            color_secondary = color_secondary.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape: (batch_size, 6)
            
            # Połącz oba kolory w jeden tensor (batch_size, 6)
            targets = torch.cat([color_primary, color_secondary], dim=1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader):.4f}')

    # Zapis modelu
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main()