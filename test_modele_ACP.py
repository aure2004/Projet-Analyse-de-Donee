import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# Colonnes sélectionnées pour l'apprentissage
selected_columns = [
    "accommodates", "bedrooms", "beds", "room_type", "bathrooms",
    "cleaning_fee", "review_scores_rating", "instant_bookable",
    "bed_type", "cancellation_policy", "property_type"
]

# Chargement des données
airbnb = pd.read_csv("airbnb_train.csv")

# Vérification que la colonne log_price existe, sinon la créer
if "log_price" not in airbnb.columns:
    airbnb["log_price"] = np.log1p(airbnb["price"])

# Séparation des features et de la cible
X = airbnb[selected_columns].copy()
y = airbnb["log_price"]

# Identification des colonnes numériques et catégoriques
numerical_columns = ["accommodates", "bedrooms", "beds", "bathrooms", "review_scores_rating"]
categorical_columns = list(set(selected_columns) - set(numerical_columns))

# Conversion des colonnes catégorielles en chaînes de caractères
X[categorical_columns] = X[categorical_columns].astype(str)

# Pipeline de prétraitement
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Ajout de l'ACP (95% de la variance expliquée)
pca = PCA(n_components=0.95)

# Pipeline complet
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),
    ('regressor', LinearRegression())
])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model.fit(X_train, y_train)

# Évaluation
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

print("=== Évaluation sur l'ensemble d'entraînement ===")
evaluate_model(y_train, model.predict(X_train))

print("\n=== Évaluation sur l'ensemble de test ===")
evaluate_model(y_test, model.predict(X_test))

# Prédictions finales
airbnb_test = pd.read_csv("airbnb_test.csv")
final_X_test = airbnb_test[selected_columns].copy()
final_X_test[categorical_columns] = final_X_test[categorical_columns].astype(str)
y_final_prediction = model.predict(final_X_test)

# Sauvegarde
prediction_example = pd.read_csv("prediction_example.csv")
prediction_example["logpred"] = y_final_prediction
prediction_example.to_csv("MaPredictionFinale.csv", index=False)
print("\nFichier de prédictions sauvegardé sous le nom 'MaPredictionFinale.csv'.")

# Vérification conformité
def estConforme(monFichier_csv):
    votre_prediction = pd.read_csv(monFichier_csv)
    fichier_exemple = pd.read_csv("prediction_example.csv")
    assert votre_prediction.columns[1] == fichier_exemple.columns[1], \
        f"Votre colonne de prédiction doit s'appeler {fichier_exemple.columns[1]}"
    assert len(votre_prediction) == len(fichier_exemple), \
        f"Vous devriez avoir {len(fichier_exemple)} prédictions"
    assert np.all(votre_prediction.iloc[:, 0] == fichier_exemple.iloc[:, 0])
    print("Fichier conforme!")

estConforme("MaPredictionFinale.csv")
