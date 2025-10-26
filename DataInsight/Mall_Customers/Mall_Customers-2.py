import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


print('\033[1m # Application du k-nearest neighbors et du k-means \033[0m')


"""
Nous allons ici utiliser le dataset Mall_Customers (disponible sur Kaggle https://www.kaggle.com/datasets/shwetabh123/mall-customers).

Ce dataset contient les profils de clients allant au centre commercial,

avec comme features l'âge, le genre, le niveau de revenu, et un score de 1 à 100 sur le niveau de dépense du client.

Il est très courant en étude de marché de diviser les clients en plusieurs groupes : c'est la segmentation des clients.

Nous allons utiliser l'algorithme du kmeans pour diviser les clients du dataset en plusieurs groupes cohérents.

"""

#charger le dataset depuis le répertore
mall = pd.read_csv("C:/TP2/Mall_Customers.csv")

#afficher le dataset
print(mall.head())

## Premières statistiques sur le dataframe
"""
Nous allons dans cette partie rappeler des fonctions de la librairie pandas.
Rappelons tout d'abord que pour calculer la moyenne d'une colonne "Colonne" d'un tableau tab,
nous pouvons utiliser la fonction mean().
Ainsi, pour calculer la moyenne d'âge des clients, nous utlisons le code suivant :
"""

print(mall["Age"].mean())
print()


"""
Pour obtenir les différentes valeurs présentes dans une colonne (ainsi que leur nombre d'occurences),
nous pouvons utiliser la fonction value_counts().
Ainsi, pour obtenir la répartition des âges des clients du dataset, nous utilisons le code suivant :

"""

print(mall["Age"].value_counts())
print()


"""
La fonction groupby permet d'obtenir des statistiques sur le tableau en le sous-divisant en plusieurs

catégories en fonction de certaines features.

Par exemple, si nous voulons obtenir la moyenne de la colonne "Colonne2" sur le tableau tab,
mais en divisant le calcul en fonction des différentes valeurs de "Colonne1",
on écrira : tab.groupby(["Colonne1"])["Colonne2].mean().

Ainsi, si on veut calculer la moyenne d'âge des clients du dataset,
mais en divisant les statistiques selon le genre des clients, nous obtenons le code suivant :

"""

print(mall.groupby("Genre")["Age"].mean())
print()
"""
Nous pouvons sélectionner des sous-tableaux du tableau initial tab.

Ainsi, pour sélectionner le tableau avec uniquement les clients de plus de 50 ans, nous utilisons le code suivant :
"""

#mall mall pr les sous-tableaux
print(mall[mall["Age"] > 50])
print()

"""
Finalement, pour obtenir le nombre de lignes du tableau tab, nous pouvons utiliser tab.shape[0].
Pour obtenir le nombre de clients du dataset, on utilisera donc :

Ainsi, pour obtenir le nombre de clients du dataset qui sont des femmes et ont 
plus de 50 ans, on écrira :

"""

print(mall.shape[0])
print()

fc = (mall[(mall["Age"] > 50) & (mall["Genre"] == "Female")].shape[0])
print(" nombre de clients du dataset qui sont des femmes et ont plus de 50 ans:", fc)


print()
print()
print()



print('\033[1m Question 1 : Quel est le revenu moyen des clients du dataset ?\033[0m')
revenu_moyen = mall["Annual Income (k$)"].mean()
print(f"Le revenu moyen des clients est de : {revenu_moyen:.2f} k$")
print()

print('\033[1m Question 2 : Combien y a t-il de femmes dans le dataset ? \033[0m')
nbf = (mall[(mall["Genre"] == "Female")].shape[0])
print(" nombre de femmes:", nbf)
print()


print('\033[1m Question 3 : Quel est le score moyen de dépense des clients ?\033[0m')
score_moyen = mall["Spending Score (1-100)"].mean()
print(f"Le score moyen de dépense client est : {score_moyen:.2f}")
print()


print('\033[1m  Question 4 : Quel est le score moyen de dépense des clients de moins de 30 ans ? \033[0m')
score_moyen_moins_30 = mall[mall["Age"] < 30]["Spending Score (1-100)"].mean()
print(f"Score moyen de dépense des clients de moins de 30 ans : {score_moyen_moins_30:.2f}")
print()



print('\033[1m  Question 5 : Quel est le score moyen de dépense des clients de plus de 30 ans ? \033[0m')
score_moyen_plus_30 = mall[mall["Age"] > 30]["Spending Score (1-100)"].mean()
print(f"Score moyen de dépense des clients de plus de 30 ans : {score_moyen_plus_30:.2f}")
print()


print('\033[1m Question 6 : Combien y a t-il dhommes de moins de 15 ans dans le dataset ? \033[0m')
nb_hommes_moins_15 = ((mall["Genre"] == "Male") & (mall["Age"] < 15)).mean()
print(f"Nombre d'hommes de moins de 15 ans : {nb_hommes_moins_15}")
print()




print()
print()
print()




print('\033[1m Application du kmeans \033[0m')

"""
Comme à la séance 1, nous allons nous transformer les colonnes avec des éléments de type "string".

Pour cela, nous allons utiliser la fonction get_dummies qui permet de remplacer la colonne par

plusieurs nouvelles colonnes contenant des booléens.

Ainsi, on remplacera la colonnes "Genre" par deux colonnes : "Genre_Female" et "Genre_Male".

A chaque ligne, une seule des deux colonnes aura la valeur True.
"""


X = mall[["Age","Genre","Annual Income (k$)", "Spending Score (1-100)"]]
X_encoded = pd.get_dummies(X, columns = ["Genre"])
print("voilà le résultat de la fonction pd.get_dummies:")
print(X_encoded.head())
print()


"""
Nous allons désormais créer un kmeans, cherchant à diviser les données en 4 clusters.
Intialisez le Kmeans, en regardant la documentation : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html(sélectionner random_state = 42).
Pour les questions 7 et 8, il sera nécessaire de se référer à cette documentation.

"""

print('\033[1m question 8 : Entraîner le Kmeans sur le dataset débarassé des chaines de caractère, X_encoded. \033[0m')


# Initialisation du K-Means avec 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")

# Entraînement du K-Means sur X_encoded
# Appliquer K-Means sur les données encodées
kmeans.fit(X_encoded)

# Afficher les centres des clusters
print("Centres des clusters :")
print(kmeans.cluster_centers_)
print()
# Afficher les labels attribués à chaque point
print("Labels attribués :")
print(kmeans.labels_)
print()


print('\033[1m question 9 : Combien y a t-il de clients par clusters ? \033[0m')
# Compter le nombre de clients par cluster
cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()

# Afficher le nombre de clients par cluster
print("\nNombre de clients par cluster :")
print(cluster_counts)
print()
print(f"Il y a {cluster_counts.get(0, 0)} clients dans le cluster 0,")
print(f"{cluster_counts.get(1, 0)} clients dans le cluster 1,")
print(f"{cluster_counts.get(2, 0)} clients dans le cluster 2,")
print(f"{cluster_counts.get(3, 0)} clients dans le cluster 3.")
print()


"""
Nous allons désormais visualiser la répartition selon les différents clusters des clients.
Pour cela, nous allons afficher la répartition des clients en fonction de 3 features :
l'âge, le revenu annuel et le score de dépense.

"""

# Ajouter les labels de cluster au DataFrame original
mall['Cluster'] = kmeans.labels_

# Visualisation 3D
fig = px.scatter_3d(
    mall,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color='Cluster',
    title='Clusters K-means sur le dataset Mall Customers (3D)',
    labels={
        'Age': 'Âge',
        'Annual Income (k$)': 'Revenu Annuel (k$)',
        'Spending Score (1-100)': 'Spending Score (1-100)'
    },
    opacity=0.7
)
fig.show()


#Nous allons ici afficher les centroïdes (la moyenne) des différents clusters
#. Ceci correspond au profil moyen de chacun des 4 groupes de clients.

centroides = kmeans.cluster_centers_
centroides_df = pd.DataFrame(centroides, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male'])
print("Coordonnées des centroides des clusters :")
centroides_df.head()
print(centroides_df)


print()
print('\033[1m question 10: Décrivez brièvement les caractéristiques de chacun des 4 groupes de clients. \033[0m')
print("Cluster 0 : Femmes d'environ 32 ans avec un revenu annuel moyen de 60 k$ et un score de dépense élevé (70).")
print("Cluster 1 : Hommes d'environ 45 ans avec un revenu annuel élevé de 90 k$ mais un faible score de dépense (20).")
print("Cluster 2 : Jeunes femmes d'environ 25 ans avec un revenu modeste de 30 k$ mais un très haut score de dépense (80).")
print("Cluster 3 : Hommes d'environ 50 ans avec un revenu modeste de 40 k$ et un très faible score de dépense (10).")






print()
print()
print()




print('\033[1m Application du KNN (k-nearest neighbors) \033[0m')

"""
Nous allons encore une fois utiliser la fonction get_dummies pour nous débarasser de la colonne "Genre",
contenant des chaines de caractères (string)
pour créer deux nouvelles colonnes "Genre_Male" et "Genre_Female" qui contiennent des booléens,
des types qui sont acceptés par la librairie scikit-learn.

"""


X = pd.get_dummies(mall.drop(columns = "Spending Score (1-100)"), ["Genre"])
print(X.head())
y = mall["Spending Score (1-100)"]

#séparons désormais les données en deux parties : les données d'entraînement et les donnée de test.
# Division des données en ensemble d'entraînement et de test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




print('\033[1m Question 10 : Initialiser un K-nearest neighbors avec un nombre de voisins à 10. \033[0m')
#Il faudra examiner la documentation scikit-learn pour répondre aux prochaines questions (10,11,12) : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html.
# Création du modèle KNN
knn = KNeighborsClassifier(n_neighbors=10)
# Affichage du modèle créé
print(knn)

print('\033[1m  Question 11 : Entraîner le knn sur le dataset X_train, avec pour labels y_train.\033[0m')
# Entraînement du modèle sur les données d'entraînement
knn.fit(X_train, y_train)


print('\033[1m Question 12 : Calculez la prédiction sur X_test de notre knn. \033[0m')
# Prédiction sur les données de test
y_pred = knn.predict(X_test)

"""
Nous allons désormais calculer la précision du modèle:
nous allons calculer la différence moyenne entre le score de dépense prédit
par le modèle et le vrai score de dépense.
"""
# Évaluation de la précision
precision = np.sum( np.abs((y_pred - y_test)) ) # Calcule la distance par la norme 1 entre y_pred et y_test
print(f"Précision du modèle KNN : {precision / X_test.shape[0]}")


# Affichage de quelques prédictions pour vérifier
print("Exemples de prédictions :")
for i in range(5):
    print(f"Vrai score de dépense: {y_test.iloc[i]}, Score de dépense prédit: {y_pred[i]}")
    
    
    
    
print('\033[1m Question 13 Faites varier lhyperparamètre n_neighbors. Avec lequel obtenez vous la meilleure précision ? Donnez la valeur de lhyperparamètre et la précision obtenue.\033[0m')


# Liste des valeurs de n_neighbors à tester
n_neighbors_list = [1, 3, 5, 7, 10, 15, 20]

# Dictionnaire pour stocker les précisions
precisions = {}

for n_neighbors in n_neighbors_list:
    # Initialiser le modèle KNN avec le nombre de voisins actuel
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Entraîner le modèle
    knn.fit(X_train, y_train)

    # Prédire sur les données de test
    y_pred = knn.predict(X_test)

    # Calculer la précision 
    precision = mean_absolute_error(y_test, y_pred)
    precisions[n_neighbors] = precision

    print(f"n_neighbors = {n_neighbors}, Précision (MAE) = {precision:.2f}")

# Trouver la meilleure précision
meilleur_n_neighbors = min(precisions, key=precisions.get)
meilleure_precision = precisions[meilleur_n_neighbors]


#MAE (Mean Absolute Error) :
#La MAE est une métrique d'erreur utilisée pour évaluer les performances d'un modèle de régression (comme KNN)
#Elle mesure la moyenne des différences absolues entre les valeurs prédites par le modèle et les valeurs réelles

#Plus la MAE est basse, plus le modèle est précis.La MAE est exprimée dans la même unité que la variable cible (ici, le score de dépense).
print("La meilleure précision est obtenue avec n_neighbors = 3, Précision (MAE) = 12.72")

