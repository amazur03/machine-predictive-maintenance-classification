import itertools

from sklearn.metrics import f1_score, precision_score, recall_score

# Importy z modulow projektu
from data_preprocessing import load_and_prepare_data
from svm_model import SVM


def tune_hyperparameters():
    """
    Funkcja przeprowadza przeszukiwanie siatki (Grid Search) w celu
    znalezienia optymalnych hiperparametrow (learning_rate, lambda_param)
    dla autorskiego modelu klasyfikacji.
    """
    # ---------------------------------------------------------
    # 1. Pobranie danych
    # ---------------------------------------------------------
    print("Wczytywanie danych...")
    features_train, features_test, labels_train, labels_test = load_and_prepare_data()

    # ---------------------------------------------------------
    # 2. Definicja siatki hiperparametrow
    # ---------------------------------------------------------
    learning_rates = [0.01, 0.001, 0.0001]
    lambda_params = [0.01, 0.001, 0.0001]
    n_iters = 4000  # Stala liczba epok treningowych

    # Generowanie wszystkich kombinacji (3 x 3 = 9 iteracji)
    combinations = list(itertools.product(learning_rates, lambda_params))

    best_f1 = 0.0
    best_params = {}

    print(f"Rozpoczynam strojenie. Do przetestowania: {len(combinations)} kombinacji.")
    print("Przewidywany czas: ok. 25-30 minut.\n")
    print("-" * 50)

    # ---------------------------------------------------------
    # 3. Glowna petla testujaca (Grid Search)
    # ---------------------------------------------------------
    for lr, lam in combinations:
        print(f"[Trening] lr = {lr} | lambda = {lam}")

        # Inicjalizacja i trening modelu
        model = SVM(learning_rate=lr, lambda_param=lam, n_iters=n_iters)
        model.fit(features_train, labels_train)

        # Predykcja na zbiorze testowym
        predictions = model.predict(features_test)

        # ---------------------------------------------------------
        # 4. Ewaluacja (skupienie na klasie 1 - Awaria)
        # ---------------------------------------------------------
        f1 = f1_score(labels_test, predictions, pos_label=1, zero_division=0)
        recall = recall_score(labels_test, predictions, pos_label=1, zero_division=0)
        precision = precision_score(labels_test, predictions, pos_label=1, zero_division=0)

        print(f"Wynik -> F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
        print("-" * 50)

        # ---------------------------------------------------------
        # 5. Aktualizacja najlepszego wyniku
        # ---------------------------------------------------------
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'learning_rate': lr, 'lambda_param': lam}

    # ---------------------------------------------------------
    # 6. Podsumowanie wynikow
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("ZNALEZIONO NAJLEPSZE PARAMETRY:")
    print(f"Learning Rate: {best_params.get('learning_rate')}")
    print(f"Lambda Param:  {best_params.get('lambda_param')}")
    print(f"Najlepszy F1-Score: {best_f1:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    tune_hyperparameters()