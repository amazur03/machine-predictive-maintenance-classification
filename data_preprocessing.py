import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import Standarization


def load_and_prepare_data():
    """
    Glowna funkcja przygotowujaca dane.
    Wykonuje pelny pipeline obrobki danych krok po kroku:
    1. Pobiera dane z platformy Kaggle.
    2. Usuwa zbedne identyfikatory i potencjalne zrodla wycieku danych.
    3. Koduje zmienne kategoryczne (One-Hot Encoding).
    4. Tworzy nowe, zaawansowane cechy fizyczne (Feature Engineering).
    5. Dzieli zbior na treningowy i testowy ze stratyfikacja.
    6. Standaryzuje dane.

    Zwraca:
    Gotowe macierze numpy: X_train, X_test, y_train, y_test.
    """

    print("Pobieranie danych...")
    path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
    file_path = os.path.join(path, "predictive_maintenance.csv")
    data = pd.read_csv(file_path)
    print("Dane wczytane:", data.shape)

    print("Czyszczenie danych...")
    # Usuniecie kolumn identyfikacyjnych oraz 'Failure Type' (zapobieganie data leakage)
    data = data.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)

    # Zamiana zmiennej kategorycznej na numeryczna (typ L pozostaje bazowy gdy M i H sa rowne 0)
    data['Type_M'] = (data['Type'] == 'M').astype(int)
    data['Type_H'] = (data['Type'] == 'H').astype(int)
    data = data.drop(columns=['Type'])

    print("Tworzenie nowych cech...")
    # Inzynieria cech: tworzenie nowych relacji matematycznych miedzy czujnikami
    data['Temp_diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['Torque_per_speed'] = data['Torque [Nm]'] / (data['Rotational speed [rpm]'] + 1)
    data['Power_estimate'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']
    data['Wear_squared'] = data['Tool wear [min]'] ** 2
    data['Torque_squared'] = data['Torque [Nm]'] ** 2
    data['Speed_squared'] = data['Rotational speed [rpm]'] ** 2
    data['Torque_x_wear'] = data['Torque [Nm]'] * data['Tool wear [min]']
    data['Speed_x_wear'] = data['Rotational speed [rpm]'] * data['Tool wear [min]']

    # Usuniecie temperatury powietrza (zastapiona przez roznice temperatur)
    data = data.drop(['Air temperature [K]'], axis=1)
    print("Liczba kolumn po feature engineering:", data.shape[1])

    print("Podzial danych...")
    features = data.drop(columns=['Target'])
    labels = data['Target']

    # Podzial danych z zachowaniem proporcji klas (stratify=labels)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

    # Konwersja formatu Pandas na macierze NumPy (wymagane do obliczen w algorytmie SVM)
    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)
    y_train = y_train.values.astype(int)
    y_test = y_test.values.astype(int)

    print("Standaryzacja...")
    # Zastosowanie autorskiej standaryzacji, ktora "uczy sie" tylko na zbiorze treningowym
    standarizator = Standarization()
    X_train = standarizator.fit_transform(X_train)
    X_test = standarizator.transform(X_test)

    return X_train, X_test, y_train, y_test