import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import scipy.stats as stats

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

# Réduction de la cardinalité des colonnes catégoriques
def reduce_cardinality(df, column, threshold=50):
    value_counts = df[column].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_categories, "Other")
    return df

airbnb = reduce_cardinality(airbnb, "city", threshold=50)
airbnb = reduce_cardinality(airbnb, "property_type", threshold=50)

# Sélection des colonnes
selected_columns = [
    "accommodates", "bedrooms", "beds", "bed_type", "room_type", "bathrooms",
    "cleaning_fee", "city", "review_scores_rating", "instant_bookable",
    "cancellation_policy", "property_type",
] + amenity_columns

# Création de nouvelles variables combinées
airbnb["accommodates_per_bedroom"] = airbnb["accommodates"] / (airbnb["bedrooms"] + 1e-6)
airbnb["bathrooms_per_bedroom"] = airbnb["bathrooms"] / (airbnb["bedrooms"] + 1e-6)
airbnb["log_bedrooms"] = np.log1p(airbnb["bedrooms"])
airbnb["log_bathrooms"] = np.log1p(airbnb["bathrooms"])
airbnb["bed_bath_ratio"] = airbnb["beds"] / (airbnb["bathrooms"] + 1e-6)
airbnb["bedroom_bathroom_product"] = airbnb["bedrooms"] * airbnb["bathrooms"]

X = airbnb[selected_columns + [
    "accommodates_per_bedroom", "bathrooms_per_bedroom", "log_bedrooms",
    "log_bathrooms", "bed_bath_ratio", "bedroom_bathroom_product"
]]
y = airbnb["log_price"]

# Séparation des colonnes numériques / catégoriques
numerical_columns = [
    "accommodates", "bedrooms", "beds", "bathrooms", "review_scores_rating",
    "accommodates_per_bedroom", "bathrooms_per_bedroom", "log_bedrooms",
    "log_bathrooms", "bed_bath_ratio", "bedroom_bathroom_product"
] + amenity_columns
categorical_columns = list(set(selected_columns) - set(numerical_columns))

# Pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Modèle XGBoost avec optimisation des hyperparamètres
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1))
])

param_dist = {
    "regressor__n_estimators": stats.randint(100, 500),
    "regressor__max_depth": stats.randint(3, 12),
    "regressor__learning_rate": stats.uniform(0.01, 0.3),
    "regressor__subsample": stats.uniform(0.6, 0.4),
    "regressor__colsample_bytree": stats.uniform(0.6, 0.4),
    "regressor__reg_alpha": stats.uniform(0, 1),
    "regressor__reg_lambda": stats.uniform(0, 1)
}

search = RandomizedSearchCV(
    model, param_distributions=param_dist,
    n_iter=50, cv=KFold(10, shuffle=True, random_state=42),
    scoring="r2", random_state=42, n_jobs=-1, verbose=1
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement avec RandomizedSearchCV
search.fit(X_train, y_train)
print("Meilleurs paramètres :", search.best_params_)

# Évaluation
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

print("\n=== Évaluation sur l'ensemble d'entraînement ===")
evaluate_model(y_train, search.predict(X_train))

print("\n=== Évaluation sur l'ensemble de test ===")
evaluate_model(y_test, search.predict(X_test))

# Prédictions finales
airbnb_test = pd.read_csv("airbnb_test.csv")
airbnb_test, _ = process_amenities_column(airbnb_test, top_n=20)
airbnb_test = reduce_cardinality(airbnb_test, "city", threshold=50)
airbnb_test = reduce_cardinality(airbnb_test, "property_type", threshold=50)
airbnb_test["accommodates_per_bedroom"] = airbnb_test["accommodates"] / (airbnb_test["bedrooms"] + 1e-6)
airbnb_test["bathrooms_per_bedroom"] = airbnb_test["bathrooms"] / (airbnb_test["bedrooms"] + 1e-6)
airbnb_test["log_bedrooms"] = np.log1p(airbnb_test["bedrooms"])
airbnb_test["log_bathrooms"] = np.log1p(airbnb_test["bathrooms"])
airbnb_test["bed_bath_ratio"] = airbnb_test["beds"] / (airbnb_test["bathrooms"] + 1e-6)
airbnb_test["bedroom_bathroom_product"] = airbnb_test["bedrooms"] * airbnb_test["bathrooms"]

final_X_test = airbnb_test[selected_columns + [
    "accommodates_per_bedroom", "bathrooms_per_bedroom", "log_bedrooms",
    "log_bathrooms", "bed_bath_ratio", "bedroom_bathroom_product"
]].copy()
final_X_test[categorical_columns] = final_X_test[categorical_columns].astype(str)
y_final_prediction = search.predict(final_X_test)

# Sauvegarde des prédictions
prediction_example = pd.read_csv("prediction_example.csv")
prediction_example["logpred"] = y_final_prediction
prediction_example.to_csv("MaPredictionFinale.csv", index=False)
print("\nFichier de prédictions sauvegardé sous le nom 'MaPredictionFinale.csv'.")