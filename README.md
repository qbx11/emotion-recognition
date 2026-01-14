
# Emotion Recognition using PyTorch (FER2013)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

Projekt implementujący konwolucyjną sieć neuronową (CNN) do rozpoznawania emocji na twarzy przy użyciu biblioteki **PyTorch**. Model trenowany jest na popularnym zbiorze danych **FER2013** (Facial Expression Recognition 2013).

##Główne cechy projektu

* **Własna architektura CNN:** 4 bloki konwolucyjne (Conv2D + BatchNorm + ReLU + MaxPool).
* **Focal Loss:** Zastosowanie niestandardowej funkcji straty (`gamma=2.0`) w celu poradzenia sobie z nierównowagą klas.
* **Data Augmentation:** Rozszerzenie zbioru treningowego poprzez losowe odbicia, rotacje, skalowanie i zmiany jasności/kontrastu.
* **Analiza wag:** Wizualizacja rozkładu wag w poszczególnych warstwach modelu.
* **Metryki:** Raporty klasyfikacji (Precision, Recall, F1-score) oraz Macierz Pomyłek (Confusion Matrix).

##Dataset

Wykorzystano zbiór **FER2013**.
* **Preprocessing:** Obrazy są skalowane do rozmiaru 48x48 pikseli (skala szarości).
* **Filtrowanie:** Ze względu na znikomą liczbę próbek i niską jakość, klasa **"Disgust"** została usunięta ze zbioru.
* **Podział:** Zbiór podzielono na: Train (80% z reszty), Validation (10%), Test (10%).




##Wymagane biblioteki:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm pillow


# Real-Time Emotion Recognition
Aplikacja oparta na sztucznej inteligencji, która rozpoznaje emocje twarzy w czasie rzeczywistym przy użyciu kamery internetowej. Projekt wykorzystuje **PyTorch** do klasyfikacji emocji oraz **MediaPipe** do szybkiego i precyzyjnego wykrywania twarzy.

### Wykrywane emocje:
* 😠  
* 😨 
* 😄 
* 😥 
* 😲 
* 😐 

##Wymagane biblioteki:

```bash
pip install torch torchvision opencv-python mediapipe numpy pillow
