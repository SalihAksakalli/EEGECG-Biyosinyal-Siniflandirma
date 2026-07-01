import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

from gradcam_explainability import make_gradcam_heatmap

def main():
    train_df = pd.read_csv("data/mitbih_train.csv", header=None)
    test_df = pd.read_csv("data/mitbih_test.csv", header=None)
    
    label_col = train_df.shape[1] - 1
    X_train = train_df.iloc[:, :label_col].values
    y_train = train_df.iloc[:, label_col].values.astype(int)
    X_test = test_df.iloc[:, :label_col].values
    y_test = test_df.iloc[:, label_col].values.astype(int)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_res), size=25000, replace=False)
    X_train_cnn = X_res[sample_indices].reshape(-1, 187, 1)
    y_train_cnn_cat = to_categorical(y_res[sample_indices], num_classes=5)
    
    model = Sequential([
        Input(shape=(187, 1)),
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train_cnn_cat, epochs=5, batch_size=256, verbose=0)
    
    last_conv = [l.name for l in model.layers if isinstance(l, Conv1D)][-1]
    
    print("\n--- Grad-CAM Değerlerinin Yoğunlaştığı İndeksler (İlk 5 ve Maksimumlar) ---")
    class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    
    X_test_cnn = X_test.reshape(-1, 187, 1)
    y_pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
    
    for cls in range(5):
        correct_idx = np.where((y_test == cls) & (y_pred == cls))[0][0]
        signal = X_test[correct_idx]
        cam = make_gradcam_heatmap(model, signal.reshape(1, 187, 1), last_conv, pred_index=cls)
        heatmap = np.interp(np.linspace(0, len(cam)-1, 187), np.arange(len(cam)), cam)
        
        # En yüksek 5 önem derecesine sahip indeksleri bul
        top_indices = np.argsort(heatmap)[::-1][:10]
        print(f"Sınıf {cls} ({class_names[cls]}):")
        print(f"  En yüksek öneme sahip ilk 10 indeks: {top_indices}")
        print(f"  Bu indekslerdeki sinyal genlikleri : {np.round(signal[top_indices], 3)}")
        print(f"  Isı haritası değerleri             : {np.round(heatmap[top_indices], 3)}")

if __name__ == "__main__":
    main()
