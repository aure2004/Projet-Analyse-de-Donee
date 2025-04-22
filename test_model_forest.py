import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score



# Chargement des données
airbnb = pd.read_csv("airbnb_train.csv")

# Création de log_price si absent
if "log_price" not in airbnb.columns:
    airbnb["log_price"] = np.log1p(airbnb["price"])

# Traitement des amenities
def process_amenities_column(df, top_n=20):
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("[]")
    all_amenities = df["amenities"].str.replace(r"[{}\"]", "", regex=True).str.split(",")
    amenities_flat = [item.strip() for sublist in all_amenities for item in sublist]
    top_amenities = pd.Series(amenities_flat).value_counts().head(top_n).index.tolist()
    for amenity in top_amenities:
        df[f"amenity_{amenity}"] = all_amenities.apply(lambda x: int(amenity in x))
    return df, [f"amenity_{a}" for a in top_amenities]

# Application à l'entraînement
airbnb, amenity_columns = process_amenities_column(airbnb, top_n=20)

# Sélection des colonnes
selected_columns = [
    "accommodates", "bedrooms", "beds", "bed_type", "room_type", "bathrooms",
    "cleaning_fee", "city", "review_scores_rating", "instant_bookable",
    "cancellation_policy", "property_type",
] + amenity_columns

X = airbnb[selected_columns]
y = airbnb["log_price"]

# Séparation des colonnes numériques / catégoriques
numerical_columns = ["accommodates", "bedrooms", "beds", "bathrooms", "review_scores_rating"] + amenity_columns
categorical_columns = list(set(selected_columns) - set(numerical_columns))

# Pipelines
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
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,  # Limite la profondeur des arbres
        min_samples_split=10,  # Minimum d'échantillons pour diviser un nœud
        min_samples_leaf=5,  # Minimum d'échantillons par feuille
        random_state=42
    ))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Scores de validation croisée : {cv_scores}")
print(f"R² moyen (validation croisée) : {np.mean(cv_scores):.4f}")


# Entraînement
model.fit(X_train, y_train)

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

# Prédictions finales
airbnb_test = pd.read_csv("airbnb_test.csv")
airbnb_test, _ = process_amenities_column(airbnb_test, top_n=20)
final_X_test = airbnb_test[selected_columns].copy()
final_X_test[categorical_columns] = final_X_test[categorical_columns].astype(str)
y_final_prediction = model.predict(final_X_test)

# Sauvegarde des prédictions
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
