# DataInsight
Travail de pre-processing pour l'usage de modèles de données en machine learning

# DataInsight: Analyse et Segmentation de Données

**Projets d'analyse de données et de machine learning** utilisant les datasets **Titanic** et **Mall Customers** pour prédire la survie des passagers et segmenter les clients.


Ce dépôt, contient deux projets d'analyse de données, fait l'usage des bibliothèques scikit-learn pour le machine learning réalisés en Python. Ces projets m'ont permis de développer des compétences en prétraitement des données, modélisation, et visualisation.

##  Projets

### 1. Prédiction de la Survie sur le Titanic

#### Description
Ce projet utilise le dataset **Titanic** pour prédire la survie des passagers. L'objectif est d'analyser les caractéristiques des passagers (âge, sexe, classe, etc.) et de construire un modèle de machine learning pour prédire leur survie.

#### Compétences Acquises
- Prétraitement des données : gestion des valeurs manquantes, encodage des variables catégorielles
- Modélisation : utilisation de `RandomForestClassifier` et `DecisionTreeClassifier`
- Évaluation des modèles : calcul de l'accuracy et de l'erreur moyenne absolue (MAE)
- Visualisation des données : utilisation de `plotly` pour visualiser les données

#### Résultats
- **Accuracy** : Environ 82% avec un modèle Random Forest
- **Feature Importance** : Le sexe, puis la classe des passagers sont les caractéristiques les plus déterminantes pour la survie


### 2. Segmentation des données clients d'un centre commercial

#### Description
Ce projet utilise le dataset **Mall Customers** pour segmenter les clients en groupes cohérents en utilisant l'algorithme K-Means. L'objectif est de comprendre les différents profils de clients en fonction de leur âge, revenu annuel, et score de dépense.

#### Compétences
- Prétraitement des données : encodage des variables catégorielles et normalisation des données
- Clustering : utilisation de `KMeans` pour segmenter les clients en 4 clusters
- Visualisation 3D : utilisation de `plotly` pour visualiser les clusters en 3D
- Évaluation des clusters : analyse des centroïdes pour décrire les caractéristiques de chaque cluster
- Modélisation avec KNN : utilisation de `KNeighborsClassifier` pour prédire le score de dépense des clients


##  Technologies

- **Langage** : Python
- **Plateforme**: Jupyter, Thonny
- **Bibliothèques** :
  - `pandas` pour la manipulation des données
  - `scikit-learn` pour le machine learning
  - `plotly` pour la visualisation des données
  - `numpy` pour les calculs numériques

#### Résultats
- **Clusters identifiés** :
  - Cluster 0 : Femmes d'environ 45 ans avec un revenu annuel moyen et un score de dépense élevé
  - Cluster 1 : Clients avec un revenu annuel élevé mais un faible score de dépense
  - Cluster 2 : Jeunes clients avec un revenu modeste mais un score de dépense élevé
  - Cluster 3 : Clients plus âgés avec un revenu modeste et un faible score de dépense
- **identification du meilleur modèle KNN K-nearest neighbors

