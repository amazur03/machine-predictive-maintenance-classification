from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importy z wlasnych modulow projektu
from data_preprocessing import load_and_prepare_data
from svm_model import SVM
from utils import plot_confusion


def main():
    """
    Glowna funkcja
    """

    # Etap 1: Przygotowanie danych
    # Wywolanie pipelinu przetwarzania danych (pobieranie, czyszczenie, inzynieria cech, podzial i standaryzacja).
    features_train, features_test, labels_train, labels_test = load_and_prepare_data()

    # Etap 2: Trening i ewaluacja autorskiego modelu SVM
    model = SVM()
    model.fit(features_train, labels_train)
    pred_svm = model.predict(features_test)

    print("\nSVM (wlasna implementacja)")
    print(classification_report(labels_test, pred_svm, zero_division=0))
    plot_confusion(labels_test, pred_svm, "Autorski SVM")

    # Etap 3: Porownanie z modelami referencyjnymi (Baseline)
    # Wykorzystujemy gotowe implementacje z biblioteki Scikit-learn z ustawionym balansem klas dla obiektywnego porownania wynikow.
    print("\nPorownanie z modelami sklearn")

    models = {
        "SVM_sklearn": SVC(class_weight='balanced'),
        "RandomForest": RandomForestClassifier(class_weight='balanced'),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced')
    }

    for name, model_sklearn in models.items():
        print("\n" + name)
        model_sklearn.fit(features_train, labels_train)
        predictions = model_sklearn.predict(features_test)

        # Wyswietlenie metryk i wygenerowanie macierzy pomylek dla kazdego modelu
        print(classification_report(labels_test, predictions, zero_division=0))
        plot_confusion(labels_test, predictions, name)

    print("\nKONIEC")


if __name__ == "__main__":
    main()