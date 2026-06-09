import time

import numpy as np


class SVM:
    """
    Liniowy klasyfikator Support Vector Machine z regularyzacja L2.
    Zaimplementowany od podstaw z obsluga niezbalansowanych klas
    oraz wlasna funkcja kosztu (Hinge Loss).
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.0001, n_iters=4000):
        """
        Inicjalizacja hiperparametrow modelu.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.loss_history = []  # Lista do zapisu historii funkcji kosztu

        # Pola inicjalizowane podczas treningu
        self.w = None
        self.b = None
        self.class_weights = None

    def fit(self, X, y):
        """
        Proces trenowania modelu na danych wejsciowych (Stochastic Gradient Descent).
        """
        print("Trenowanie autorskiego SVM...")
        start_time = time.time()

        # ---------------------------------------------------------
        # 1. Przygotowanie danych i wag klas
        # ---------------------------------------------------------
        y_converted = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        n_neg = np.sum(y_converted == -1)
        n_pos = np.sum(y_converted == 1)

        # Wagi odwrotnie proporcjonalne do liczebnosci klas
        weight_n = n_samples / (2 * n_neg)
        weight_p = n_samples / (2 * n_pos)
        self.class_weights = {-1: weight_n, 1: weight_p}

        print(f"  Waga klasy brak awarii: {weight_n:.4f}")
        print(f"  Waga klasy awarii:      {weight_p:.4f}")

        # ---------------------------------------------------------
        # 2. Inicjalizacja parametrow
        # ---------------------------------------------------------
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []  # Wyzerowanie historii przed nowym treningiem

        # ---------------------------------------------------------
        # 3. Glowna petla uczaca
        # ---------------------------------------------------------
        for epoch in range(self.n_iters):
            # Losowe przetasowanie indeksow (SGD)
            indices = np.random.permutation(n_samples)

            # Wygaszanie wspolczynnika uczenia (Learning Rate Decay)
            lr = self.learning_rate / (1 + epoch * 0.005)

            for idx in indices:
                x_i = X[idx]
                y_i = y_converted[idx]
                weight = self.class_weights[y_i]

                # Sprawdzenie warunku marginesu
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Prawidlowa klasyfikacja (tylko krok dla L2)
                    self.w -= lr * (2 * self.lambda_param * self.w)
                else:
                    # Bledna klasyfikacja (krok dla L2 + Hinge Loss uwzgledniajacy wage)
                    self.w -= lr * (2 * self.lambda_param * self.w - weight * y_i * x_i)
                    self.b += lr * (weight * y_i)

            # ---------------------------------------------------------
            # 4. Obliczanie funkcji kosztu na koniec epoki
            # ---------------------------------------------------------
            distances = 1 - y_converted * (np.dot(X, self.w) + self.b)
            distances[distances < 0] = 0  # Odpowiednik max(0, distance)

            weights_vector = np.where(y_converted == 1, weight_p, weight_n)

            hinge_loss = np.mean(weights_vector * distances)
            l2_reg = self.lambda_param * np.dot(self.w, self.w)
            total_loss = hinge_loss + l2_reg

            self.loss_history.append(total_loss)

            # Raportowanie postepow co 500 epok
            if epoch % 500 == 0:
                print(f"Epoka {epoch:<4} - Koszt: {total_loss:.4f}")

        elapsed_time = time.time() - start_time
        print(f"Trening zakonczony w {elapsed_time:.2f} s")

    def predict(self, X):
        """
        Przewidywanie etykiet dla nowych danych na podstawie wyuczonych wag.
        Zwraca 1 dla klasy pozytywnej i 0 dla klasy negatywnej.
        """
        result = np.dot(X, self.w) + self.b
        return np.where(result >= 0, 1, 0)