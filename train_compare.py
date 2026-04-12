import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from data_utils import CWRUDatasetTorch, build_svm_dataset
from models_ext import SimpleCNN, ResNet18Classifier


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = BASE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS_CNN = 20
EPOCHS_RESNET = 20
LR = 0.001


def train_torch_model(model, train_loader, test_loader, epochs, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses = []
    test_accs = []
    best_acc = 0.0
    best_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accs.append(test_acc)
        
        print(f"[{model_name}] Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Test Acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            
    elapsed = time.time() - start_time
    
    # Calculate final confusion matrix with best model or current model? 
    # Usually we want confusion matrix of the best model, but loading it back might be slow.
    # Let's use current model for simplicity, or load best. Let's load best.
    model.load_state_dict(torch.load(best_path))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot curves immediately
    plot_curves(train_losses, test_accs, model_name)
    plot_confusion_matrix(cm, model_name)
    
    return {
        "model_name": model_name,
        "best_acc": best_acc,
        "train_losses": train_losses,
        "test_accs": test_accs,
        "confusion_matrix": cm,
        "best_path": best_path,
        "elapsed": elapsed
    }


def plot_curves(train_losses, test_accs, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker="o")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accs, marker="s")
    plt.title(f"{model_name} Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_curves.png"))
    plt.close()


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ["Normal", "Inner", "Outer", "Ball"], rotation=45)
    plt.yticks(tick_marks, ["Normal", "Inner", "Outer", "Ball"])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_confusion.png"))
    plt.close()


def plot_svm_curve(model, X, y):
    train_sizes = np.linspace(0.1, 1.0, 5)
    sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, scoring="accuracy", n_jobs=1, shuffle=True, random_state=42
    )
    train_mean = train_scores.mean(axis=1) * 100
    test_mean = test_scores.mean(axis=1) * 100
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, train_mean, marker="o", label="Train Acc")
    plt.plot(sizes, test_mean, marker="s", label="Test Acc")
    plt.xlabel("Train Samples")
    plt.ylabel("Accuracy (%)")
    plt.title("SVM Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "SVM_curve.png"))
    plt.close()


def train_svm():
    print("[SVM] Training...")
    X, y = build_svm_dataset(DATA_DIR)
    X = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    start_time = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    elapsed = time.time() - start_time
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "SVM")
    plot_svm_curve(svm, X, y)
    
    print(f"[SVM] Test Acc={acc:.2f}%")
    
    return {
        "model_name": "SVM",
        "best_acc": acc,
        "confusion_matrix": cm,
        "best_path": "N/A",
        "elapsed": elapsed
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading Dataset...")
    full_dataset = CWRUDatasetTorch(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    results = []

    # 2. Train SimpleCNN
    print("\n=== Training SimpleCNN ===")
    cnn = SimpleCNN(num_classes=4).to(device)
    results.append(train_torch_model(cnn, train_loader, test_loader, EPOCHS_CNN, device, "SimpleCNN"))

    # 3. Train ResNet18
    print("\n=== Training ResNet18 ===")
    resnet = ResNet18Classifier(num_classes=4).to(device)
    results.append(train_torch_model(resnet, train_loader, test_loader, EPOCHS_RESNET, device, "ResNet18"))
    
    # 4. Train SVM
    print("\n=== Training SVM ===")
    results.append(train_svm())

    # 5. Summary
    print("\n=== Comparative Summary ===")
    summary_path = os.path.join(OUTPUT_DIR, "compare_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model,Best Accuracy (%),Training Time (s),Model Path\n")
        print(f"{'Model':<15} | {'Best Acc':<10} | {'Time (s)':<10}")
        print("-" * 40)
        for r in results:
            print(f"{r['model_name']:<15} | {r['best_acc']:.2f}%      | {r['elapsed']:.2f}")
            f.write(f"{r['model_name']},{r['best_acc']:.2f},{r['elapsed']:.2f},{r['best_path']}\n")

    # Save best model meta
    best = max(results, key=lambda x: x["best_acc"])
    best_meta = {"model_name": best["model_name"], "best_acc": best["best_acc"], "best_path": best["best_path"]}
    with open(os.path.join(OUTPUT_DIR, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_meta, f, ensure_ascii=False, indent=2)
    print(f"\nBest Model: {best['model_name']} with {best['best_acc']:.2f}% accuracy.")


if __name__ == "__main__":
    main()
