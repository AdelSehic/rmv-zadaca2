#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
import os

# CIFAR-10 class names
class_names = ['Avion', 'Automobil', 'Ptica', 'Mačka', 'Jelen', 
               'Pas', 'Žaba', 'Konj', 'Brod', 'Kamion']

def unpickle(file):
    """Učitavanje CIFAR-10 batch fajlova"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(cifar_path="cifar-10-batches-py"):
    """Učitavanje CIFAR-10 podataka iz postojeće lokacije"""
    if not os.path.exists(cifar_path):
        print(f"Greška: CIFAR-10 dataset nije pronađen na putanji: {cifar_path}")
        print("Molimo unesite ispravnu putanju do CIFAR-10 dataseta.")
        cifar_path = input("Unesite putanju do cifar-10-batches-py folder: ")
        
        if not os.path.exists(cifar_path):
            raise FileNotFoundError(f"CIFAR-10 dataset nije pronađen na: {cifar_path}")
    
    print(f"Učitavanje CIFAR-10 dataseta iz: {cifar_path}")
    
    # Učitavanje training batches
    x_train = []
    y_train = []
    
    for i in range(1, 6):
        batch_path = os.path.join(cifar_path, f'data_batch_{i}')
        batch = unpickle(batch_path)
        x_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])
    
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    
    # Učitavanje test batch
    test_batch_path = os.path.join(cifar_path, 'test_batch')
    test_batch = unpickle(test_batch_path)
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # Reshape slike (32x32x3 = 3072 piksela)
    x_train = x_train.reshape(50000, 32, 32, 3)
    x_test = x_test.reshape(10000, 32, 32, 3)
    
    print(f"Učitano {len(x_train)} training slika i {len(x_test)} test slika")
    
    return (x_train, y_train), (x_test, y_test)

def extract_features(images):
    """Ekstraktovanje jednostavnih feature-a iz slika"""
    n_samples = images.shape[0]
    
    # Flatten slike
    flattened = images.reshape(n_samples, -1)
    
    # Normalizacija
    flattened = flattened.astype('float32') / 255.0
    
    # Dodatni features
    features = []
    
    for img in images:
        # Srednje vrednosti po kanalima
        mean_r = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 2])
        
        # Standardne devijacije po kanalima
        std_r = np.std(img[:, :, 0])
        std_g = np.std(img[:, :, 1])
        std_b = np.std(img[:, :, 2])
        
        # Histogram features (pojednostavljeno)
        hist_r = np.histogram(img[:, :, 0], bins=8)[0]
        hist_g = np.histogram(img[:, :, 1], bins=8)[0]
        hist_b = np.histogram(img[:, :, 2], bins=8)[0]
        
        feature_vector = np.concatenate([
            [mean_r, mean_g, mean_b, std_r, std_g, std_b],
            hist_r, hist_g, hist_b
        ])
        
        features.append(feature_vector)
    
    features = np.array(features)
    
    # Kombinovanje sa flattened slikama (uzimamo samo deo zbog memorije)
    combined = np.concatenate([flattened[:, ::8], features], axis=1)  # Svaki 8. piksel
    
    return combined

def main():
    print("=== CIFAR-10 KLASIFIKACIJA SA SCIKIT-LEARN ===\n")
    
    # 1. Učitavanje podataka
    print("1. Učitavanje CIFAR-10 dataseta...")
    
    cifar_path = os.path.join(os.path.curdir, "neural_network_sets", "cifar10")
    
    (x_train, y_train), (x_test, y_test) = load_cifar10(cifar_path)
    
    print(f"Training set: {x_train.shape}")
    print(f"Test set: {x_test.shape}")
    print(f"Klase: {len(class_names)}\n")
    
    # 2. Priprema podataka
    print("2. Ekstraktovanje feature-a...")
    
    # Koristimo manji subset za brže testiranje
    train_subset = 20000
    test_subset = 2000
    
    x_train_subset = x_train[:train_subset]
    y_train_subset = y_train[:train_subset]
    x_test_subset = x_test[:test_subset]
    y_test_subset = y_test[:test_subset]
    
    # Ekstraktovanje features
    x_train_features = extract_features(x_train_subset)
    x_test_features = extract_features(x_test_subset)
    
    print(f"Feature shape: {x_train_features.shape}")
    
    # 3. Skaliranje podataka
    print("3. Skaliranje podataka...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_features)
    x_test_scaled = scaler.transform(x_test_features)
    
    # 4. PCA za smanjenje dimenzionalnosti
    print("4. Primena PCA...")
    pca = PCA(n_components=100)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    
    print(f"PCA variance ratio: {pca.explained_variance_ratio_[:5]}")
    
    # 5. Treniranje različitih modela
    print("\n5. Treniranje modela...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTreniranje {name}...")
        
        if name == 'SVM':
            # SVM sa manjim subset-om zbog brzine
            train_size = 1000
            model.fit(x_train_pca[:train_size], y_train_subset[:train_size])
        else:
            model.fit(x_train_pca, y_train_subset)
        
        # Predikcije
        y_pred = model.predict(x_test_pca)
        accuracy = accuracy_score(y_test_subset, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} tačnost: {accuracy:.4f}")
    
    # 6. Detaljne metrike za najbolji model
    print("\n6. Detaljne metrike...")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]
    
    print(f"\nNajbolji model: {best_model_name}")
    print(f"Tačnost: {best_model['accuracy']:.4f}")
    
    print(f"\nClassification Report za {best_model_name}:")
    print(classification_report(y_test_subset, best_model['predictions'], 
                              target_names=class_names))
    
    # 7. Vizualizacija
    print("\n7. Kreiranje vizualizacija...")
    
    # Rezultati modela
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bar plot tačnosti
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title('Tačnost različitih modela')
    axes[0, 0].set_ylabel('Tačnost')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_subset, best_model['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predviđena klasa')
    axes[0, 1].set_ylabel('Stvarna klasa')
    
    # PCA komponente
    axes[1, 0].plot(pca.explained_variance_ratio_[:20])
    axes[1, 0].set_title('PCA - Explained Variance Ratio')
    axes[1, 0].set_xlabel('Komponenta')
    axes[1, 0].set_ylabel('Variance Ratio')
    axes[1, 0].grid(True)
    
    # Tačnost po klasama
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    axes[1, 1].bar(class_names, class_accuracy)
    axes[1, 1].set_title('Tačnost po klasama')
    axes[1, 1].set_xlabel('Klasa')
    axes[1, 1].set_ylabel('Tačnost')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cifar10_sklearn_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Prikaz primera
    print("\n8. Prikaz primera predikcija...")
    
    # Random uzorci
    np.random.seed(42)
    sample_indices = np.random.choice(len(x_test_subset), 12, replace=False)
    
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(sample_indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_test_subset[idx])
        
        true_label = class_names[y_test_subset[idx]]
        pred_label = class_names[best_model['predictions'][idx]]
        
        color = 'green' if y_test_subset[idx] == best_model['predictions'][idx] else 'red'
        plt.title(f'Tačno: {true_label}\nPred: {pred_label}', 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_sklearn_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== ANALIZA ZAVRŠENA ===")
    print(f"Najbolji model: {best_model_name}")
    print(f"Tačnost: {best_model['accuracy']:.2%}")
    print("Rezultati su sačuvani kao slike.")

if __name__ == "__main__":
    main()
