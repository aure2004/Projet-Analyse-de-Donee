import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def data_processing_and_viz(filepath):
    # 1. Chargement du jeu de données
    data = pd.read_csv(filepath)

    # 2. Prix / log-prix
    if 'log_price' in data.columns:
        data['price'] = np.expm1(data['log_price'])
    else:
        data['log_price'] = np.log1p(data['price'])

    # 3. Aperçu rapide
    print(f"Jeu de données : {data.shape[0]} lignes × {data.shape[1]} colonnes")
    print("\nValeurs manquantes par colonne :")
    print(data.isnull().sum())

    # 4. Histogrammes du prix
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(data['price'].dropna(), bins=40, edgecolor='black')
    plt.title('Prix brut')
    plt.xlabel('Prix (€)')
    plt.ylabel('Effectif')
    plt.subplot(1,2,2)
    plt.hist(data['log_price'].dropna(), bins=40, edgecolor='black')
    plt.title('Log-prix')
    plt.xlabel('Log(prix)')
    plt.ylabel('Effectif')
    plt.tight_layout()
    plt.show()

    # 5. Extraction manuelle des amenities et encodage binaire
    #    On retire les accolades & guillemets, on split sur la virgule
    raw = (data['amenities']
           .fillna('[]')
           .str.replace(r'[\{\}"]', '', regex=True)
           .str.split(',')
           .apply(lambda lst: [x.strip() for x in lst if x.strip()]))
    flat = pd.Series([item for sublist in raw for item in sublist])
    top_25 = flat.value_counts().head(25).index.tolist()
    for amenity in top_25:
        col_name = 'has_' + amenity.replace(' ', '_')
        data[col_name] = raw.apply(lambda lst: int(amenity in lst))

    # 6. Regroupement des petites villes
    city_counts = data['city'].value_counts()
    small = city_counts[city_counts < 75].index
    data['city_cat'] = data['city'].where(~data['city'].isin(small), 'Other')

    # 7. Création de features supplémentaires
    data['price_per_guest']   = data['price'] / data['accommodates']
    data['beds_per_bedroom']  = data['beds'] / (data['bedrooms'] + 1e-6)
    data['bath_per_bed_ratio'] = data['bathrooms'] / (data['beds'] + 1e-6)

    # 8. Clustering géographique (5 clusters)
    coords = data[['latitude','longitude']].dropna()
    scaler = StandardScaler()
    coords_norm = scaler.fit_transform(coords)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(coords_norm)
    data.loc[coords.index, 'geo_group'] = kmeans.labels_

    # 9. Visualisation des clusters
    plt.figure(figsize=(6,6))
    for grp in sorted(data['geo_group'].unique()):
        subset = data[data['geo_group'] == grp]
        plt.scatter(
            subset['longitude'], subset['latitude'],
            s=10, alpha=0.6,
            label=f'Cluster {grp}'
        )
    plt.legend()
    plt.title('Groupes géographiques')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # 10. Carte de corrélation des variables numériques
    features = [
        'accommodates','bedrooms','beds','bathrooms','review_scores_rating',
        'price_per_guest','beds_per_bedroom','bath_per_bed_ratio','log_price'
    ]
    corr_matrix = data[features].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr_matrix, interpolation='nearest')
    plt.colorbar(shrink=0.8)
    plt.xticks(range(len(features)), features, rotation=90)
    plt.yticks(range(len(features)), features)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.show()

    return data

if __name__ == "__main__":
    df = data_processing_and_viz('airbnb_train.csv')
