import torch
import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_and_plot_metrics(model, test_loader, device, output_dir):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    # VISUALISASI CONFUSION MATRIX
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal/Pre (0)', 'Hipertensi (1)', 'Hyp II (2)'], 
                yticklabels=['Normal/Pre (0)', 'Hipertensi (1)', 'Hyp II (2)'],
                annot_kws={"size": 16})
    plt.xlabel('Prediksi AI', fontsize=12)
    plt.ylabel('Label Faktual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    # Simpan gambar matriks
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"-> Gambar Confusion Matrix disimpan ke {cm_path}")
    
    # Gunakan output_dict=True agar hasil berbentuk dictionary
    report_dict = classification_report(
        all_targets, all_preds, 
        target_names=['Normal/Pre (0)', 'Hipertensi (1)', 'Hyp II (2)'], 
        output_dict=True
    )
    
    # Hitung Specificity dan tambahkan ke dictionary
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        report_dict['specificity'] = {"Normal Class": specificity}
        
    # Simpan ke file JSON
    json_path = os.path.join(output_dir, "classification_report.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"-> Laporan JSON berhasil disimpan ke {json_path}")

    # VISUALISASI TABEL LAPORAN (.png)
    # Hapus kunci 'specificity' sementara agar struktur tabel scikit-learn rapi
    report_for_table = {k: v for k, v in report_dict.items() if k != 'specificity'}
    
    # Ubah dictionary ke Pandas DataFrame agar mudah digambar
    df_report = pd.DataFrame(report_for_table).transpose()
    df_report = df_report.round(4) # Bulatkan angka 4 desimal di belakang koma
    
    # Buat kanvas kosong untuk menggambar tabel
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.axis('tight')
    
    # Gambar tabel
    table = ax.table(cellText=df_report.values, rowLabels=df_report.index, colLabels=df_report.columns, cellLoc='center',loc='center')
    
    # Modifikasi tampilan tabel 
    table.scale(1, 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Warna latar belakang biru pada header kolom dan baris
    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
            
    plt.title('Classification Report', fontsize=15, pad=20, weight='bold')
    
    # Simpan gambar tabel
    table_path = os.path.join(output_dir, "classification_report_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"-> Gambar Tabel Laporan disimpan ke {table_path}")

def plot_training_results(train_losses, val_losses, train_accs, val_accs, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Cross Entropy Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-o', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Model Accuracy')

    plt.tight_layout()
    # Simpan gambar grafik training
    chart_path = os.path.join(output_dir, "training_results.png")
    plt.savefig(chart_path, dpi=300)
    plt.show()
    print(f"-> Grafik Training disimpan ke {chart_path}")