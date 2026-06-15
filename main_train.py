import os
import torch
from src.dataset import get_classification_loaders
from src.model import CNN_LSTM_Classifier
from src.engine import train_model
from src.evaluate import evaluate_and_plot_metrics, plot_training_results

if __name__ == "__main__":
    # Path dataset input
    DATA_PATH = "data/raw" 
    
    # PATH OUTPUT SPESIFIK KE GOOGLE DRIVE
    DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/Dataset_Magang/outputs"
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- MENGGUNAKAN DEVICE: {device.type.upper()} ---\n")
    print(f"--- FOLDER OUTPUT: {DRIVE_OUTPUT_DIR} ---\n")
    
    # 1. LOAD DATA
    print("--- 1. Memuat dan Memproses Data ---")
    train_loader, val_loader, test_loader, class_weights = get_classification_loaders(
        DATA_PATH, batch_size=16, val_split=0.15, test_split=0.15
    )
    
    # 2. INIT MODEL
    print("\n--- 2. Inisialisasi Model AI Klasifikasi ---")
    model = CNN_LSTM_Classifier().to(device)
    
    # 3. TRAINING
    print("\n--- 3. Memulai Proses Training ---")
    train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, val_loader, device, epochs=50, lr=1e-4, class_weights=class_weights
    )
    
    # 4. PLOT TRAINING
    print("\n--- 4. Membuat Grafik Performa Training ---")
    plot_training_results(train_loss, val_loss, train_acc, val_acc, output_dir=DRIVE_OUTPUT_DIR)
    
    # 5. TESTING & METRICS
    print("\n--- 5. Fase Testing & Matriks Evaluasi ---")
    # Load model yang disimpan ke Drive
    save_path = os.path.join(DRIVE_OUTPUT_DIR, "bp_classifier_model.pth")
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    evaluate_and_plot_metrics(model, test_loader, device, output_dir=DRIVE_OUTPUT_DIR)