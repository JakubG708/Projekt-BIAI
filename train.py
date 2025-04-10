import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import ColorDataset
from model import NeuralNet

# Konfiguracja
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 30
MODEL_PATH = 'color_predictor.pth'

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Trening
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, colors in dataloader:
            inputs, colors = inputs.to(device), colors.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, colors)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader):.4f}')

    # Zapis modelu
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main()