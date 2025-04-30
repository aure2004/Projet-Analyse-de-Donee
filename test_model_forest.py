import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import scipy.stats as stats

# 1. Chargement des données d'entraînement
train_df = pd.read_csv("airbnb_train.csv")
if "log_price" not in train_df.columns:
    raise KeyError("La colonne 'log_price' est manquante dans airbnb_train.csv")

# 2. Extraction des amenities (top 20)
def process_amenities(df, top_n=20):
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("[]")
    cleaned = (
        df["amenities"]
        .str.replace(r"[{}\"]", "", regex=True)
        .str.split(",")
        .apply(lambda lst: [x.strip() for x in lst if x.strip()])
    )
    flat = [a for sub in cleaned for a in sub]
    top = pd.Series(flat).value_counts().head(top_n).index.tolist()
    for amen in top:
        df[f"amenity_{amen}"] = cleaned.apply(lambda x: int(amen in x))
    return df, [f"amenity_{amen}" for amen in top]

train_df, amenity_cols = process_amenities(train_df, top_n=20)

# 3. Sélection des features de base
base_feats = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "cleaning_fee", "review_scores_rating",
    "instant_bookable", "cancellation_policy",
    "room_type", "property_type", "city"
]

# 3.1 Création de nouvelles features combinées
train_df["accommodates_per_bathroom"] = train_df["accommodates"] / (train_df["bathrooms"] + 1e-6)
train_df["bedrooms_per_accommodates"] = train_df["bedrooms"] / (train_df["accommodates"] + 1e-6)
train_df["beds_per_bedroom"] = train_df["beds"] / (train_df["bedrooms"] + 1e-6)
train_df["price_per_accommodate"] = train_df["cleaning_fee"] / (train_df["accommodates"] + 1e-6)
train_df["bathrooms_per_bedroom"] = train_df["bathrooms"] / (train_df["bedrooms"] + 1e-6)
train_df["log_bedrooms"] = np.log1p(train_df["bedrooms"])
train_df["log_accommodates"] = np.log1p(train_df["accommodates"])
train_df["log_bathrooms"] = np.log1p(train_df["bathrooms"])
train_df["cleaning_fee_indicator"] = (train_df["cleaning_fee"] > 0).astype(int)
train_df["has_reviews"] = (train_df["review_scores_rating"] > 0).astype(int)
train_df["is_shared_room"] = (train_df["room_type"] == "Shared room").astype(int)

# Regrouper les catégories rares dans 'city'
rare_cities = train_df["city"].value_counts()[train_df["city"].value_counts() < 50].index
train_df["city"] = train_df["city"].replace(rare_cities, "Other")

# Ajouter les nouvelles colonnes aux features
combined_feats = [
    "accommodates_per_bathroom", "bedrooms_per_accommodates",
    "beds_per_bedroom", "price_per_accommodate", "bathrooms_per_bedroom",
    "log_bedrooms", "log_accommodates", "log_bathrooms",
    "cleaning_fee_indicator", "has_reviews", "is_shared_room"
]
features = base_feats + amenity_cols + combined_feats
target = "log_price"

# 4. Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df[features], train_df[target], test_size=0.2, random_state=42
)

# 5. Construction du préprocesseur
num_feats = [col for col in features if train_df[col].dtype in ["int64", "float64"]]
cat_feats = [col for col in features if col not in num_feats]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_feats),
    ("cat", cat_pipe, cat_feats)
])

# 6. Pipeline XGBoost avec RandomizedSearchCV
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])
param_dist = {
    "model__n_estimators": stats.randint(100, 400),
    "model__max_depth": stats.randint(3, 12),
    "model__learning_rate": stats.uniform(0.01, 0.3),
    "model__subsample": stats.uniform(0.6, 0.4),
    "model__colsample_bytree": stats.uniform(0.6, 0.4),
    "model__reg_alpha": stats.uniform(0, 1),
    "model__reg_lambda": stats.uniform(0, 1)
}
search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=15, cv=3, scoring="r2",
    random_state=42, n_jobs=-1, verbose=1
)

# 7. Entraînement et hyperparam tuning
search.fit(X_train, y_train)
print("Meilleurs paramètres :", search.best_params_)

# 8. Évaluation du modèle
for name, (X_set, y_set) in zip(["Train", "Validation"], [(X_train, y_train), (X_val, y_val)]):
    preds = search.predict(X_set)
    rmse = np.sqrt(mean_squared_error(y_set, preds))
    r2 = r2_score(y_set, preds)
    print(f"{name} — R²: {r2:.4f}, RMSE: {rmse:.4f}")

# 9. Prédiction finale
test_df = pd.read_csv("airbnb_test.csv")
test_df, _ = process_amenities(test_df, top_n=20)

# Appliquer les mêmes transformations sur test_df
test_df["accommodates_per_bathroom"] = test_df["accommodates"] / (test_df["bathrooms"] + 1e-6)
test_df["bedrooms_per_accommodates"] = test_df["bedrooms"] / (test_df["accommodates"] + 1e-6)
test_df["beds_per_bedroom"] = test_df["beds"] / (test_df["bedrooms"] + 1e-6)
test_df["price_per_accommodate"] = test_df["cleaning_fee"] / (test_df["accommodates"] + 1e-6)
test_df["bathrooms_per_bedroom"] = test_df["bathrooms"] / (test_df["bedrooms"] + 1e-6)
test_df["log_bedrooms"] = np.log1p(test_df["bedrooms"])
test_df["log_accommodates"] = np.log1p(test_df["accommodates"])
test_df["log_bathrooms"] = np.log1p(test_df["bathrooms"])
test_df["cleaning_fee_indicator"] = (test_df["cleaning_fee"] > 0).astype(int)
test_df["has_reviews"] = (test_df["review_scores_rating"] > 0).astype(int)
test_df["is_shared_room"] = (test_df["room_type"] == "Shared room").astype(int)
test_df["city"] = test_df["city"].replace(rare_cities, "Other")

# Préparer les données pour la prédiction
X_test = test_df[features].copy()
X_test[cat_feats] = X_test[cat_feats].astype(str)
preds_final = search.predict(X_test)

submission = pd.read_csv("prediction_example.csv")
submission[submission.columns[1]] = preds_final
submission.to_csv("MaPredictionFinale.csv", index=False)
print("MaPredictionFinale.csv généré !")

# 10. Vérification de conformité
def estConforme(fpath):
    pred = pd.read_csv(fpath)
    example = pd.read_csv("prediction_example.csv")
    assert pred.columns[1] == example.columns[1]
    assert len(pred) == len(example)
    assert np.all(pred.iloc[:, 0] == example.iloc[:, 0])
    print("Fichier conforme !")

estConforme("MaPredictionFinale.csv")