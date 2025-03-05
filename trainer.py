import torch
import torch.nn.functional as F
def train(optimizer, train_loader, model, criterion, epochs, valid_loader, writer):
    best_loss=float('inf')
    best_model_weights=None
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print("train_loss: ", avg_loss, "epoch: ", epoch)
        validation_running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_running_loss += loss.item()
                if loss.item() <= best_loss:
                    best_loss = loss.item()
                    best_model_weights = model.state_dict().copy()
        avg_validation_loss = validation_running_loss / len(valid_loader)
        writer.add_scalar('Loss/validation', avg_validation_loss, epoch)
        print("validation_loss: ", avg_validation_loss, "epoch: ", epoch)
    torch.save(best_model_weights, 'best_model_weights.pth')
def test(model, test_loader, writer):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs=model(images)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy: ", accuracy)
