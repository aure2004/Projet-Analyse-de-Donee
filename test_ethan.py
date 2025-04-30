import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import scipy.stats as stats

# 1. Chargement des données
train_df = pd.read_csv("airbnb_train.csv")
test_df = pd.read_csv("airbnb_test.csv")

# Création de log_price si absent
if "log_price" not in train_df.columns:
    train_df["log_price"] = np.log1p(train_df["price"])

# 2. Exploration des données
print("Dimensions :", train_df.shape)
print("\nTypes de colonnes :\n", train_df.dtypes.value_counts())
print("\nValeurs manquantes :\n", train_df.isnull().sum())

# 3. Traitement des amenities
def process_amenities(df, top_n=20):
    df = df.copy()
    df["amenities"] = df["amenities"].fillna("[]")
    all_amenities = df["amenities"].str.replace(r"[{}\"]", "", regex=True).str.split(",")
    amenities_flat = [item.strip() for sublist in all_amenities for item in sublist]
    top_amenities = pd.Series(amenities_flat).value_counts().head(top_n).index.tolist()
    for amenity in top_amenities:
        df[f"amenity_{amenity}"] = all_amenities.apply(lambda x: int(amenity in x))
    return df, [f"amenity_{a}" for a in top_amenities]

train_df, amenity_columns = process_amenities(train_df, top_n=20)

# 4. Réduction de la cardinalité des colonnes catégoriques
def reduce_cardinality(df, column, threshold=50):
    value_counts = df[column].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_categories, "Other")
    return df

train_df = reduce_cardinality(train_df, "city", threshold=50)
train_df = reduce_cardinality(train_df, "property_type", threshold=50)

# 5. Création de nouvelles variables
train_df["accommodates_per_bedroom"] = train_df["accommodates"] / (train_df["bedrooms"] + 1e-6)
train_df["bathrooms_per_bedroom"] = train_df["bathrooms"] / (train_df["bedrooms"] + 1e-6)
train_df["log_bedrooms"] = np.log1p(train_df["bedrooms"])
train_df["log_bathrooms"] = np.log1p(train_df["bathrooms"])
train_df["bed_bath_ratio"] = train_df["beds"] / (train_df["bathrooms"] + 1e-6)
train_df["bedroom_bathroom_product"] = train_df["bedrooms"] * train_df["bathrooms"]

# 6. Sélection des colonnes
selected_columns = [
    "accommodates", "bedrooms", "beds", "bathrooms", "cleaning_fee",
    "review_scores_rating", "instant_bookable", "cancellation_policy",
    "room_type", "property_type", "city"
] + amenity_columns + [
    "accommodates_per_bedroom", "bathrooms_per_bedroom", "log_bedrooms",
    "log_bathrooms", "bed_bath_ratio", "bedroom_bathroom_product"
]

X = train_df[selected_columns]
y = train_df["log_price"]

# 7. Prétraitement des données
num_feats = [col for col in selected_columns if train_df[col].dtype in ["int64", "float64"]]
cat_feats = [col for col in selected_columns if col not in num_feats]

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

# 8. Modèle XGBoost avec RandomizedSearchCV
pipeline = Pipeline([
    ("preproc", preprocessor),
    ("model", XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])
param_dist = {
    "model__n_estimators": stats.randint(100, 500),
    "model__max_depth": stats.randint(3, 12),
    "model__learning_rate": stats.uniform(0.01, 0.3),
    "model__subsample": stats.uniform(0.6, 0.4),
    "model__colsample_bytree": stats.uniform(0.6, 0.4),
    "model__reg_alpha": stats.uniform(0, 1),
    "model__reg_lambda": stats.uniform(0, 1)
}
search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=50, cv=KFold(10, shuffle=True, random_state=42),
    scoring="r2", random_state=42, n_jobs=-1, verbose=1
)

# 9. Entraînement et évaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
search.fit(X_train, y_train)
print("Meilleurs paramètres :", search.best_params_)

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

print("\n=== Évaluation sur l'ensemble d'entraînement ===")
evaluate_model(y_train, search.predict(X_train))

print("\n=== Évaluation sur l'ensemble de validation ===")
evaluate_model(y_val, search.predict(X_val))

# 10. Prédictions finales
test_df, _ = process_amenities(test_df, top_n=20)
test_df = reduce_cardinality(test_df, "city", threshold=50)
test_df = reduce_cardinality(test_df, "property_type", threshold=50)
test_df["accommodates_per_bedroom"] = test_df["accommodates"] / (test_df["bedrooms"] + 1e-6)
test_df["bathrooms_per_bedroom"] = test_df["bathrooms"] / (test_df["bedrooms"] + 1e-6)
test_df["log_bedrooms"] = np.log1p(test_df["bedrooms"])
test_df["log_bathrooms"] = np.log1p(test_df["bathrooms"])
test_df["bed_bath_ratio"] = test_df["beds"] / (test_df["bathrooms"] + 1e-6)
test_df["bedroom_bathroom_product"] = test_df["bedrooms"] * test_df["bathrooms"]

X_test = test_df[selected_columns]
X_test[cat_feats] = X_test[cat_feats].astype(str)
preds_final = search.predict(X_test)

submission = pd.read_csv("prediction_example.csv")
submission[submission.columns[1]] = preds_final
submission.to_csv("MaPredictionFinale.csv", index=False)
print("MaPredictionFinale.csv généré !")