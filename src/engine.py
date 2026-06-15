import torch
import torch.nn as nn
import torch.optim as optim
import os

class EarlyStopping:
    def __init__(self, output_dir, patience=7, min_delta=0):
        self.output_dir = output_dir
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "bp_classifier_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f" -> Model terbaik ditemukan & disimpan ke Drive! (Val Loss: {self.best_loss:.4f})")

def train_model(model, train_loader, val_loader, device, output_dir, epochs=50, lr=1e-3):
    # Masukkan output_dir ke EarlyStopping
    early_stopping = EarlyStopping(output_dir=output_dir, patience=7, min_delta=0.001)

    weights = torch.tensor([1.0, 5.0, 15.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == y.data).item()
            total_train += len(y)

        # VALIDATION
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == y.data).item()
                total_val += len(y)

        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "f"Loss: [Tr {train_losses[-1]:.4f}, Val {avg_val_loss:.4f}] | "f"Acc: [Tr {train_acc*100:.2f}%, Val {val_acc*100:.2f}%]")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping aktif pada epoch {epoch+1}!")
            break

    return train_losses, val_losses, train_accs, val_accs