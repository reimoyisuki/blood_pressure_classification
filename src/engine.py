import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, class_weights=None, output_dir="outputs"):
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    # PERBAIKAN: Naikkan kesabaran (patience) menjadi 15 agar tidak berhenti terlalu cepat
    patience = 15 
    patience_counter = 0
    
    # Inisialisasi wadah metrik
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: [Tr {train_loss:.4f}, Val {val_loss:.4f}] | Acc: [Tr {train_acc:.2f}%, Val {val_acc:.2f}%]")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "bp_classifier_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping aktif pada epoch {epoch+1}!")
            break
            
    return train_losses, val_losses, train_accuracies, val_accuracies