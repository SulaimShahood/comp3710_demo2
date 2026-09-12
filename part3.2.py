import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

def get_modified_resnet18():
    """Adapts standard ResNet18 for 32x32 CIFAR-10 images."""
    model = torchvision.models.resnet18(weights=None)
    # Replace the initial 7x7 conv with a 3x3 conv and remove the initial maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running DAWNBench on: {device}")

    # Data Augmentation & Normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = get_modified_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and OneCycleLR scheduler for rapid convergence
    epochs = 24
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epochs)
    
    # Mixed Precision Scaler for speed
    scaler = torch.cuda.amp.GradScaler()

    print("Initiating Fast Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Autocast enables mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f}")
        
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    torch.save(model.state_dict(), 'dawnbench_resnet18.pth')

if __name__ == "__main__":
    main()