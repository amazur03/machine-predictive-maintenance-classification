import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Standarization:
    """
    Autorska implementacja standaryzacji danych (Z-score normalization).
    Przeksztalca cechy tak, aby mialy srednia rowna 0 i odchylenie standardowe 1.
    Klasa zostala zaprojektowana tak, aby unikac wycieku danych (data leakage)
    poprzez obliczanie i zapamietywanie parametrow wylacznie ze zbioru treningowego.
    """

    def __init__(self):
        # Inicjalizacja atrybutow klasy
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        """
        Oblicza srednia i odchylenie standardowe dla kazdej kolumny,
        a nastepnie standaryzuje podane dane (zbior treningowy).
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Zabezpieczenie numeryczne: dodanie ulamka (epsilon) zapobiega
        # bledowi dzielenia przez zero, jesli jakas cecha ma stala wartosc.
        self.std[self.std == 0] = 1e-8

        return (X - self.mean) / self.std

    def transform(self, X):
        """
        Standaryzuje nowe dane (np. testowe) uzywajac parametrow
        "nauczonych" wczesniej w metodzie fit_transform().
        """
        return (X - self.mean) / self.std


def plot_confusion(true_labels, predicted_labels, title):
    """
    Generuje i wyswietla macierz pomylek (Confusion Matrix) w formie
    czytelnej mapy cieplnej (heatmap) przy uzyciu biblioteki Seaborn.
    """
    matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(6, 5))

    # Rysowanie mapy cieplnej
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Brak awarii', 'Awaria'],
        yticklabels=['Brak awarii', 'Awaria']
    )

    # Konfiguracja wygladu wykresu
    plt.title(title)
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywiste")
    plt.tight_layout()

    plt.show()