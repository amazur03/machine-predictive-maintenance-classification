import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Importy z wlasnych modulow projektu
from data_preprocessing import load_and_prepare_data
from svm_model import SVM
from utils import plot_confusion


def plot_learning_curve(loss_history, title="Krzywa uczenia autorskiego modelu SVM"):
    """
    Generuje i wyswietla wykres krzywej uczenia dla podanego modelu.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_history)), loss_history, color='b', linewidth=2)

    # Konfiguracja wygladu wykresu
    plt.title(title)
    plt.xlabel('Epoka (Iteracja)')
    plt.ylabel('Wartosc funkcji kosztu (Hinge Loss + L2)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()


def main():
    """
    Glowna funkcja uruchamiajaca proces uczenia, ewaluacji
    oraz porownania modeli.
    """
    # ---------------------------------------------------------
    # Etap 1: Przygotowanie danych
    # ---------------------------------------------------------
    features_train, features_test, labels_train, labels_test = load_and_prepare_data()

    # ---------------------------------------------------------
    # Etap 2: Trening i ewaluacja autorskiego modelu SVM
    # ---------------------------------------------------------
    model = SVM()
    model.fit(features_train, labels_train)

    # Wizualizacja procesu uczenia po zakonczeniu treningu
    plot_learning_curve(model.loss_history)

    # Predykcja i raport dla wlasnej implementacji
    pred_svm = model.predict(features_test)

    print("\n--- SVM (wlasna implementacja) ---")
    print(classification_report(labels_test, pred_svm, zero_division=0))
    plot_confusion(labels_test, pred_svm, "Autorski SVM")

    # ---------------------------------------------------------
    # Etap 3: Porownanie z modelami referencyjnymi (Baseline)
    # ---------------------------------------------------------
    print("\n--- Porownanie z modelami sklearn ---")

    # Zbior modeli bazowych (z uwzglednieniem balansu klas)
    models = {
        "SVM_sklearn": SVC(kernel='linear', class_weight='balanced'),
        "RandomForest": RandomForestClassifier(class_weight='balanced'),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced')
    }

    # Uruchomienie, trening i ewaluacja kazdego z modeli referencyjnych
    for name, model_sklearn in models.items():
        print(f"\nModel: {name}")

        model_sklearn.fit(features_train, labels_train)
        predictions = model_sklearn.predict(features_test)

        print(classification_report(labels_test, predictions, zero_division=0))
        plot_confusion(labels_test, predictions, name)

    print("\nKONIEC")


if __name__ == "__main__":
    main()