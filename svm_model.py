import numpy as np
import time

class SVM:
    """
    Liniowy klasyfikator Support Vector Machine z regularyzacja L2.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=4000):
        """
        Inicjalizacja hiperparametrow modelu.

        Parametry:
        - learning_rate: Poczatkowy krok uczenia (szybkosc aktualizacji wag).
        - lambda_param: Wspolczynnik regularyzacji L2 (zapobiega przeuczeniu).
        - n_iters: Liczba epok (ile razy model przejdzie przez caly zbior danych).
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        """
        Proces trenowania modelu na danych wejsciowych.
        """
        print("Trenowanie autorskiego SVM...")
        start = time.time()

        # Zmiana etykiet z 0 (brak awarii) i 1 (awaria) na -1 i 1.
        # Jest to wymog matematyczny dla poprawnego dzialania funkcji Hinge Loss.
        y_converted = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Obliczanie wag klas dla problemu niezbalansowanego.
        # Klasa mniejszosciowa (awaria) otrzyma wyzsza wage,
        # co zwiekszy matematyczna kare za jej bledna klasyfikacje.
        n_neg = np.sum(y_converted == -1)
        n_pos = np.sum(y_converted == 1)

        weight_n = n_samples / (2 * n_neg)
        weight_p = n_samples / (2 * n_pos)
        self.class_weights = {-1: weight_n, 1: weight_p}

        print("  Waga klasy brak awarii:", round(weight_n, 4))
        print("  Waga klasy awarii:     ", round(weight_p, 4))

        # Inicjalizacja wag (wektor w) i wyrazu wolnego (bias b) zerami
        self.w = np.zeros(n_features)
        self.b = 0

        # Glowna petla uczaca (epoki)
        for epoch in range(self.n_iters):

            # Losowe przemieszanie probek w kazdej epoce (wlasciwosc SGD)
            indices = np.random.permutation(n_samples)

            # Dynamiczne wygaszanie kroku uczenia (Learning Rate Decay).
            # Zmniejsza krok wraz z uplywem czasu, zapewniajac stabilniejsza zbieznosc.
            lr = self.learning_rate / (1 + epoch * 0.005)

            for idx in indices:
                x_i = X[idx]
                y_i = y_converted[idx]
                weight = self.class_weights[y_i]

                # Sprawdzenie warunku marginesu (Hinge Loss)
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Probka znajduje sie po wlasciwej stronie marginesu.
                    # Stosujemy tylko kare z regularyzacji L2 (lekkie zmniejszanie wag).
                    self.w -= lr * (2 * self.lambda_param * self.w)
                else:
                    # Probka blednie sklasyfikowana lub znajduje sie wewnatrz marginesu.
                    # Korygujemy wagi w kierunku poprawnej klasyfikacji,
                    # uwzgledniajac wage danej klasy (weight).
                    self.w -= lr * (2 * self.lambda_param * self.w - weight * y_i * x_i)
                    self.b += lr * (weight * y_i)

            # Raportowanie postepu w konsoli
            if epoch % 500 == 0:
                print("Epoka", epoch)

        print("Trening zakonczony w", round(time.time() - start, 2), "s")

    def predict(self, X):
        """
        Przewidywanie etykiet dla nowych danych na podstawie wyuczonych wag.
        Zwraca 1 dla wartosci wiekszych lub rownych 0, oraz 0 w przeciwnym razie.
        """
        # Rownanie hiperpłaszczyzny decyzyjnej: w*x + b
        result = np.dot(X, self.w) + self.b
        return np.where(result >= 0, 1, 0)