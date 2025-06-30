#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils as ut

IMAGE = "lena.png"

def show_histogram(image, title="Histogram"):
    """
    Prikazuje histogram grayscale slike
    """
    # Izračunavanje histograma
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Kreiranje figure za histogram
    plt.figure(figsize=(10, 6))
    plt.plot(hist)
    plt.title(f'{title} - Histogram')
    plt.xlabel('Intenzitet piksela (0-255)')
    plt.ylabel('Broj piksela')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return hist

def create_binary_images(gray_image):
    """
    Kreira binarne slike pomoću različitih metoda
    """
    images = []
    titles = []
    
    # 1. Originalna grayscale slika
    images.append(gray_image)
    titles.append("Original Grayscale")
    
    # 2. Simple thresholding (fiksni prag)
    ret, thresh_simple = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    images.append(thresh_simple)
    titles.append(f"Simple Threshold (127)")
    
    # 3. Otsu's thresholding (automatski prag)
    ret_otsu, thresh_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images.append(thresh_otsu)
    titles.append(f"Otsu Threshold ({int(ret_otsu)})")
    
    # 4. Adaptive Mean thresholding
    thresh_adaptive_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    images.append(thresh_adaptive_mean)
    titles.append("Adaptive Mean")
    
    # 5. Adaptive Gaussian thresholding
    thresh_adaptive_gauss = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    images.append(thresh_adaptive_gauss)
    titles.append("Adaptive Gaussian")
    
    # 6. Triangle thresholding
    ret_triangle, thresh_triangle = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    images.append(thresh_triangle)
    titles.append(f"Triangle ({int(ret_triangle)})")
    
    return images, titles

def analyze_thresholding_methods():
    """
    Analizira različite metode thresholding-a sa objašnjenjima
    """
    print("\n" + "="*60)
    print("OBJAŠNJENJE METODA BINARIZACIJE")
    print("="*60)
    
    print("\n1. SIMPLE THRESHOLDING:")
    print("   - Koristi fiksni prag (npr. 127)")
    print("   - Pikseli ispod praga → 0 (crno)")
    print("   - Pikseli iznad praga → 255 (bijelo)")
    print("   - Jednostavan ali ne uvijek optimalan")
    
    print("\n2. OTSU'S THRESHOLDING:")
    print("   - Automatski pronalazi optimalni prag")
    print("   - Minimizira intra-class varijansu")
    print("   - Dobro za slike sa bimodalnim histogramom")
    print("   - Vraća izračunati prag")
    
    print("\n3. ADAPTIVE MEAN THRESHOLDING:")
    print("   - Prag se računa za svaki piksel lokalno")
    print("   - Koristi prosjek susjednih piksela")
    print("   - Dobro za slike sa promjenjivim osvjetljenjem")
    
    print("\n4. ADAPTIVE GAUSSIAN THRESHOLDING:")
    print("   - Slično adaptive mean-u")
    print("   - Koristi Gaussian-weighted prosjek")
    print("   - Bolje za slike sa gradientnim osvjetljenjem")
    
    print("\n5. TRIANGLE THRESHOLDING:")
    print("   - Koristi geometrijski pristup")
    print("   - Dobro za slike sa jednim dominantnim peak-om")
    print("   - Automatski kao Otsu")

def compare_histograms(original, binary_images, titles):
    """
    Poredi histograme originalne i binarnih slika
    """
    plt.figure(figsize=(15, 10))
    
    # Histogram originalne slike
    plt.subplot(2, 3, 1)
    hist_orig = cv2.calcHist([original], [0], None, [256], [0, 256])
    plt.plot(hist_orig)
    plt.title('Original - Histogram')
    plt.xlabel('Intenzitet')
    plt.ylabel('Broj piksela')
    plt.grid(True, alpha=0.3)
    
    # Histogrami binarnih slika (preskačemo originalnu na poziciji 0)
    for i in range(1, min(6, len(binary_images))):
        plt.subplot(2, 3, i + 1)
        hist = cv2.calcHist([binary_images[i]], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.title(f'{titles[i]} - Histogram')
        plt.xlabel('Intenzitet')
        plt.ylabel('Broj piksela')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Učitavanje slike
    img = cv2.imread(IMAGE)
    if img is None:
        print(f"Greška: Nije moguće učitati sliku {IMAGE}")
        return
    
    # Konverzija u grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ut.save_images([gray], "original", "saved/zadatak4", 400)
    
    print("ZADATAK 4: Histogram grayscale slike i binarne slike")
    print("="*55)
    
    # Prikaz originalne slike i histograma
    print("\n1. Prikazujem originalnu sliku i njen histogram...")
    show_histogram(gray, "Originalna slika")
    
    # Kreiranje binarnih slika
    print("\n2. Kreiram binarne slike različitim metodama...")
    binary_images, binary_titles = create_binary_images(gray)
    
    # Konverzija za prikaz
    display_images = ut.gray_to_bgr(binary_images)
    
    # Dodavanje naslova na slike
    ut.addImageTitles(display_images, binary_titles)

    ut.save_images(display_images, binary_titles, "saved/zadatak4", 400)
    
    # Prikaz u gridu
    ut.createGridNoTitles(3, 2, display_images)
    
    # Analiza metoda
    analyze_thresholding_methods()
    
    # Poređenje histograma
    print("\n3. Prikazujem poređenje histograma...")
    compare_histograms(gray, binary_images, binary_titles)
    
    print("\n" + "="*60)
    print("OBJAŠNJENJE HISTOGRAMA GRAYSCALE SLIKE")
    print("="*60)
    print("\nHistogram pokazuje distribuciju intenziteta piksela:")
    print("- X-osa: Intenzitet piksela (0-255)")
    print("- Y-osa: Broj piksela sa tim intenzitetom")
    print("- Tamne slike: histogram pomjeren lijevo")
    print("- Svijetle slike: histogram pomjeren desno")
    print("- Kontrastne slike: širok histogram")
    print("- Niske kontrast: uzak histogram")
    
    print("\nBinarne slike imaju histogram sa samo 2 vrha:")
    print("- Vrh na 0: crni pikseli")
    print("- Vrh na 255: bijeli pikseli")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
