import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Chargement des données
airbnb = pd.read_csv("airbnb_train.csv")

if "log_price" not in airbnb.columns:
    airbnb["log_price"] = np.log1p(airbnb["price"])

# Traitement des amenities
def process_amenities_column(df, top_n=50):
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("[]")
    all_amenities = df["amenities"].str.replace(r"[{}\"]", "", regex=True).str.split(",")
    amenities_flat = [item.strip() for sublist in all_amenities for item in sublist]
    top_amenities = pd.Series(amenities_flat).value_counts().head(top_n).index.tolist()
    for amenity in top_amenities:
        df[f"amenity_{amenity}"] = all_amenities.apply(lambda x: int(amenity in x))
    return df, [f"amenity_{a}" for a in top_amenities]

airbnb, amenity_columns = process_amenities_column(airbnb, top_n=20)

# Sélection des colonnes
selected_columns = [
    "accommodates", "bedrooms", "beds", "bed_type", "room_type", "bathrooms",
    "cleaning_fee", "city", "review_scores_rating", "instant_bookable",
    "cancellation_policy", "property_type"
] + amenity_columns

X = airbnb[selected_columns].copy()
y = airbnb["log_price"]

# Nettoyage + encodage logique
X["review_scores_rating"] = X["review_scores_rating"].fillna(X["review_scores_rating"].median())
X["bathrooms"] = pd.to_numeric(X["bathrooms"], errors='coerce').fillna(0)
X["cleaning_fee"] = X["cleaning_fee"].fillna("False").map({"True": 1, "False": 0})
X["instant_bookable"] = X["instant_bookable"].fillna("False").map({"True": 1, "False": 0})

# Feature engineering
X["room_score"] = X["accommodates"] * X["bedrooms"] * X["beds"]
X["bed_bath_ratio"] = X["bedrooms"] / (X["bathrooms"] + 1)
X["log_accommodates"] = np.log1p(X["accommodates"])
X["bedroom_beds_ratio"] = X["bedrooms"] / (X["beds"] + 1)

# Réduction du bruit sur les colonnes catégoriques
for col in ["city", "property_type"]:
    top_values = X[col].value_counts().index[:20]
    X[col] = X[col].where(X[col].isin(top_values), "other")

# Redéfinition des colonnes
numerical_columns = [
    "accommodates", "bedrooms", "beds", "bathrooms", "review_scores_rating", 
    "room_score", "bed_bath_ratio", "log_accommodates", "bedroom_beds_ratio"
] + amenity_columns
categorical_columns = list(set(selected_columns + ["room_score"]) - set(numerical_columns))

# Pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Pipeline modèle
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=150,  # Nombre d'arbres
        learning_rate=0.1,  # Vitesse d'apprentissage
        max_depth=5,  # Profondeur maximale des arbres
        min_samples_split=5,  # Nombre minimal d'échantillons pour diviser un noeud
        min_samples_leaf=2,  # Nombre minimal d'échantillons dans une feuille
        random_state=42
    ))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuster le modèle
model.fit(X_train, y_train)

# Validation croisée pour évaluer la généralisation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Scores de validation croisée : {cv_scores}")
print(f"R² moyen (validation croisée) : {np.mean(cv_scores):.4f}")

# Évaluation
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

print("\n=== Évaluation sur l'ensemble d'entraînement ===")
evaluate_model(y_train, model.predict(X_train))

print("\n=== Évaluation sur l'ensemble de test ===")
evaluate_model(y_test, model.predict(X_test))

# Test final
airbnb_test = pd.read_csv("airbnb_test.csv")
airbnb_test, _ = process_amenities_column(airbnb_test, top_n=20)
final_X_test = airbnb_test[selected_columns].copy()

# Prétraitement cohérent avec l'entraînement
final_X_test["review_scores_rating"] = final_X_test["review_scores_rating"].fillna(X["review_scores_rating"].median())
final_X_test["bathrooms"] = pd.to_numeric(final_X_test["bathrooms"], errors='coerce').fillna(0)
final_X_test["cleaning_fee"] = final_X_test["cleaning_fee"].fillna("False").map({"True": 1, "False": 0})
final_X_test["instant_bookable"] = final_X_test["instant_bookable"].fillna("False").map({"True": 1, "False": 0})
final_X_test["room_score"] = final_X_test["accommodates"] * final_X_test["bedrooms"] * final_X_test["beds"]
final_X_test["bed_bath_ratio"] = final_X_test["bedrooms"] / (final_X_test["bathrooms"] + 1)
final_X_test["log_accommodates"] = np.log1p(final_X_test["accommodates"])
final_X_test["bedroom_beds_ratio"] = final_X_test["bedrooms"] / (final_X_test["beds"] + 1)

for col in ["city", "property_type"]:
    final_X_test[col] = final_X_test[col].where(final_X_test[col].isin(X[col].unique()), "other")

final_X_test[categorical_columns] = final_X_test[categorical_columns].astype(str)

# Prédictions
y_final_prediction = model.predict(final_X_test)

# Sauvegarde
prediction_example = pd.read_csv("prediction_example.csv")
prediction_example["logpred"] = y_final_prediction
prediction_example.to_csv("MaPredictionFinale.csv", index=False)
print("\nFichier de prédictions sauvegardé sous le nom 'MaPredictionFinale.csv'.")

# Vérification
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
