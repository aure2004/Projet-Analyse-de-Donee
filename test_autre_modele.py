import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import scipy.stats as stats

# 1. Chargement des données
airbnb = pd.read_csv("airbnb_train.csv")

# 2. Création de log_price si absent
if "log_price" not in airbnb.columns:
    airbnb["log_price"] = np.log1p(airbnb["price"])

# 3. Traitement des amenities
def process_amenities_column(df, top_n=20):
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("[]")
    all_amenities = (
        df["amenities"]
        .str.replace(r"[{}\"]", "", regex=True)
        .str.split(",")
        .apply(lambda lst: [item.strip() for item in lst if item.strip()])
    )
    amenities_flat = [item for sublist in all_amenities for item in sublist]
    top_amenities = pd.Series(amenities_flat).value_counts().head(top_n).index.tolist()
    for amenity in top_amenities:
        df[f"amenity_{amenity}"] = all_amenities.apply(lambda x: int(amenity in x))
    return df, [f"amenity_{a}" for a in top_amenities]

airbnb, amenity_columns = process_amenities_column(airbnb, top_n=20)

# 4. Ajout de features optionnelles (quartier)
if "neighbourhood_cleansed" in airbnb.columns:
    airbnb["neighbourhood_cleansed"] = airbnb["neighbourhood_cleansed"].astype(str)

# 5. Sélection des colonnes\
selected_columns = [
    "accommodates", "bedrooms", "beds", "bed_type", "room_type", "bathrooms",
    "cleaning_fee", "city", "review_scores_rating", "instant_bookable",
    "cancellation_policy", "property_type", "neighbourhood_cleansed"
] + amenity_columns
X = airbnb[selected_columns]
y = airbnb["log_price"]

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Pipelines de prétraitement
numeric_feats = ["accommodates", "bedrooms", "beds", "bathrooms", "review_scores_rating"] + amenity_columns
categorical_feats = [col for col in selected_columns if col not in numeric_feats]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, categorical_feats)
])

# 8. Modèle HistGradientBoostingRegressor
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(random_state=42))
])

# 9. Recherche d'hyperparamètres avec RandomizedSearchCV
param_distributions = {
    "model__max_iter": stats.randint(100, 500),
    "model__max_depth": stats.randint(3, 12),
    "model__learning_rate": stats.uniform(0.01, 0.3),
    "model__min_samples_leaf": stats.randint(20, 100),
    "model__l2_regularization": stats.uniform(0.0, 1.0)
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 10. Entraînement de la recherche
search.fit(X_train, y_train)
print("Meilleurs paramètres :", search.best_params_)

# 11. Évaluation
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

evaluate("Train", y_train, search.predict(X_train))
evaluate("Test", y_test, search.predict(X_test))

# 12. Prédictions finales
airbnb_test = pd.read_csv("airbnb_test.csv")
airbnb_test, _ = process_amenities_column(airbnb_test, top_n=20)
if "neighbourhood_cleansed" in airbnb_test.columns:
    airbnb_test["neighbourhood_cleansed"] = airbnb_test["neighbourhood_cleansed"].astype(str)
final_X_test = airbnb_test[selected_columns].copy()
final_X_test[categorical_feats] = final_X_test[categorical_feats].astype(str)
y_pred_final = search.predict(final_X_test)

submission = pd.read_csv("prediction_example.csv")
submission[submission.columns[1]] = y_pred_final
submission.to_csv("MaPredictionFinale.csv", index=False)
print("\nFichier MaPredictionFinale.csv généré.")

# 13. Vérification de conformité
def estConforme(file):
    pred = pd.read_csv(file)
    example = pd.read_csv("prediction_example.csv")
    assert pred.columns[1] == example.columns[1], f"Colonne doit s'appeler {example.columns[1]}"
    assert len(pred) == len(example), f"Doit contenir {len(example)} lignes"
    assert np.all(pred.iloc[:,0] == example.iloc[:,0])
    print("Fichier conforme!")

estConforme("MaPredictionFinale.csv")
