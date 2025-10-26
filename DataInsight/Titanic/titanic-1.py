import pandas as pd
import numpy as np
import csv

#bibliothèques pour le preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



#https://colab.research.google.com/drive/1ogck5g9XX_eob0UPyg3Deq5zBYDIAwRm#scrollTo=SAOJH8sHO-OH


tit = pd.read_csv("C:/TP1/titanic.csv")

# Chargement du fichier titanic sous la formed'un dataframe Pandae tit
#afficher le type de la variable, ici un dataframe
type(tit);

#afficher les 5 premieres lignes du tableau
#et leur contenu
print(tit.head());

#afficher les colonnes du dataframe 
print(tit.columns);

#1ère statistiques
print(tit["Survived"].value_counts());

print()
print()

#QUESTIONS AUXQUELLES REPONDRE

#Chercher désormais la répartition de femmes et d'hommes dans les données.
#Combien y a t-il d'hommes dans le dataset ?
print(tit["Sex"].value_counts());

#afficher plus spécifique les hommes
print((tit["Sex"] == "male").sum());


print()
print()

#Cherchez la répartition des passagers parmi les différentes classes (1ère classe, 2ème classe, 3ème classe).
print(tit["Pclass"].value_counts())
#vérifier nb total de passager
print(len(tit["Pclass"]))

print()
print()

#Méthode plus précise
print(tit.groupby(["Sex", "Pclass"]) ["Survived"].mean()* 100)
#On observe d'une part que peu importe les classses, la feature sexe
#montre que les femmes ont plus survécues que les hommes, en bien plus grand nombres que les hommes
print()

print('\033[1mNous pouvons déjà observer certaines caractéristiques : comment les features Sex et pClass influencent elles les chances de survie des passagers ?\033[0m')

"""
D'après les résultats par la moyenne de survie selon le groupement par sexe et classe,  nous pouvons observer d'une part que peu importe les classses, la feature sexe montre que les femmes ont plus survécues que les hommes, en bien plus grand nombres que les hommes.
D'autre part, la feature PClass montre que ce sont les personnes membres des classes 1 qui ont le plus eut de chances de survie chez les femmes et surtout chez les hommes, avec des gros écart entre la classe 1 et les 2 autres classes (moins marquées chez les femmes) ensuite les personnes des classes 2, puis 3. 
Ca montre une forte corrélation entre le sexe, puis la classe qui ont semble jouer un role pour la survie
"""

print()

print('\033[1mCalculer le prix moyen (feature Fare) des tickets sur tous les passagers.\033[0m')

res = tit["Fare"].mean();
print(f"le prix moyen des tickets est de: {res: .2f} euro")

print()
print()

print('\033[1mCalculer le prix moyen des tickets, en divisant les passagers selon leur classe.\033[0m')

# Calculer le prix moyen des tickets par classe
prix_moyen_classes = tit.groupby("Pclass")["Fare"].mean()

# Afficher le résultat
print("Prix moyen des tickets par classe (en euros) :")
print(prix_moyen_classes)


print()
print()

print('\033[1mQuel est lâge moyen des passagers sur le bateau?\033[0m')
age_moyen = tit["Age"].mean();
print(f"Lage moyen des passagers était de: {age_moyen: .2f} ans");


print()
print()


print("Autre caractéristique des tableaux pandas : Sélection de sous-tableau")


nombre_enfants = len(tit[tit["Age"] < 15])
print(f"Nombre de passagers de moins de 15 ans : {nombre_enfants}")
print("Il y a 78 passagers de moins de 15 ans.")


print()
print()
print()
print('\033[1m# Random Forest\033[0m')
#algorithmes de machine learning 
#lalgorithme Random Forest da la librairie $scikit-learn$ pour prédire si un passager survivra ou non
#Random Forest de  scikit_learn  ne peut qu'utiliser des features float ou pouvant se transformer en float (integer, ou booléen)

"""
On cherchera à prédire, à partir de toutes les autres caractéristiques, si un passagers survivra ou non. Pour cela nous allons créer un vecteur
avec les labels (1 ou 0, pour oui ou non sur la colonne "Survived") pour chacun de nos passagers.

"""

#En machine learning, on sépare toujours :
#X : Les données utilisées pour faire la prédiction (features)
#y : La variable que l'on veut prédire (label)

y = tit["Survived"] # on isole la variable cible qu'un veut prédire "Survived"


#On crée ensuite un dataset contenant les features
X = tit.drop("Survived", axis = 1)

"""
Preprocessing
Certaines de nos features sont de type  string  comme "Embarked" ou "Name" par exemple. Cela pose problème : Random Forest de  scikit_learn  ne peut qu'utiliser des features float ou pouvant se transformer en float (integer, ou booléen).

Pour pouvoir utiliser le Random Forest de scikit_learn, nous allons d'abord nous débarasser des colonnes difficiles (pour faciliter le problème) telles que "Name" ou "Ticket".

"""


X = X.drop(["Name","Ticket"], axis = 1)

"""
Désormais, nous allons renommer les valeurs de la feature Cabin pour sélectionner uniquement la lettre du pont correspondant : le numéro de cabine "C128" devient uniquement "C".
"""

# Extrait la première lettre de chaque valeur dans la colonne "Cabin"
#Par exemple, si une cabine est notée "C123", cette ligne la transforme en "C"

X["Cabin"] = X["Cabin"].fillna('N') # Remplace les valeurs manquantes (NaN) par N
X["Cabin"] = X["Cabin"].apply(lambda x : x[0])
X["Cabin"].value_counts().head()
#on observe que bcp de cabines sont en N donc ne sont pas renseignées N


"""
Nous allons transformer les features de type  string  restantes en plusieurs colonnes contenant des booléens (car les booléens peuvent facilement se transformer en float :  True  donne 1 et  False  donne 0).

Commençons par la feature "Sex" et transformons là en deux colonnes : une "Sex_female" et une "Sex_male". Désormais, un passager aura deux features "female" et "male", avec uniquement une des deux features qui sera égale à True, et l'autre à False.

"""

#La fonction transforme la colonne en deux colonnes binaires
X_encoded = pd.get_dummies(X, columns=['Sex'])
#X_encoded est un nouveau DataFrame où la colonne "Sex" a été remplacée par Sex_female et Sex_male

#afficher les 1eres colonnes pr vérifier si les modifications de colonnes ont bien été faites:
print(X_encoded.head());


print()
print('\033[1m Utilisez la méthode des dummies pour transformer la feature "Embarked" de type  string  en plusieurs colonnes de type booléen. Attention ! Pensez bien à modifier le dataset avec nos modifications.\033[0m')
# Appliquer pd.get_dummies() sur la colonne "Embarked"
#Cette commande transforme la colonne "Embarked" en trois colonnes en booléen
X_encoded = pd.get_dummies(X_encoded, columns=['Embarked'])

#La colonne "Embarked" (qui contient "S", "C", "Q") est supprimée et 3 nvll colonnnes en booléen sont crées
print(X_encoded.head(100))

print()
print('\033[1m Faaites de même pour la feature "Cabin".\033[0m')
X_encoded = pd.get_dummies(X_encoded, columns=['Cabin'])
print(X_encoded.columns) # voir la liste des colonnes pour vérifier


#pd.set_option('display.max_columns', None)  # sinon Affiche toutes les colonnes
#print(X_encoded.head())

"""
On divise désormais nos données  en deux : les données d'entraînement, utilisées pour l'apprentissage du modèle, et les données de test, utilisées pour tester le modèle. \\

Ici nous avons $test\_size = 0.2$ donc les données de test représenteront $20\%$ des données totales, et les données d'entraînement $80\%$. \\

La valeur de random_state correspond au choix d'une clé qui génère des nombres aléatoirement.
"""

#. Division des données
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#random_state=42, mettre une "graîne aléatoire" pour que la séparation soit reproductible (toujours les mêmes données dans train et test)
#Objectif : Séparer les données en deux parties :
#80% pour l'entraînement (X_train, y_train) : Le modèle apprend sur ces données
#20% pour le test (X_test, y_test) : On utilise ces données pour évaluer la performance du modèle
#Pourquoi ? :Si on utilise les mêmes données pour entraîner et tester le modèle, on ne peut pas savoir s'il généralise bien à de nouvelles données




#Initialisons désormais un Random Forest avec 100 arbres.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#entrainer le random forest, du modèle
rf_model.fit(X_train, y_train)

#calculer la prédiction de l'algo à partir du dataset
#Utiliser le modèle entraîné pour prédire la survie (y_pred) sur les données de test (X_test)
y_pred = rf_model.predict(X_test)

#On calcule désormais l'erreur de prédiction du Random Forest :
erreur = np.mean(np.abs(y_pred-y_test))
print(f"L'erreur sur le test est de {erreur * 100:.2f}%.")

accuracy = 1 - erreur
print(f"L'accuracy sur le test est de {accuracy * 100:.2f}%.")

print()
print('\033[1m Runnez une nouvelle fois les cases avec un nombre darbres de 1 et de 10. Notez les résultats.Quelle est linfluence du nombre darbre sur la performance du Random Forest ?\033[0m')

rf_model_1 = RandomForestClassifier(n_estimators=1, random_state=42)
rf_model_1.fit(X_train, y_train)
y_pred_1 = rf_model_1.predict(X_test)
erreur_1 = np.mean(np.abs(y_pred_1 - y_test))
accuracy_1 = 1 - erreur_1
print(f"Nombre d'arbres = 1 : Erreur = {erreur_1 * 100:.2f}%, Accuracy = {accuracy_1 * 100:.2f}%")


rf_model_10 = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model_10.fit(X_train, y_train)
y_pred_10 = rf_model_10.predict(X_test)
erreur_10 = np.mean(np.abs(y_pred_10 - y_test))
accuracy_10 = 1 - erreur_10
print(f"Nombre d'arbres = 10 : Erreur = {erreur_10 * 100:.2f}%, Accuracy = {accuracy_10 * 100:.2f}%")


"""
On peut donc observer  que le nombre d'arbres a bien une influence sur l'analyse du jeux de données
Avec 1 arbre, le modèle est imprécis (taux d'erreur de 33%), et fait du sous-apprentissage ou underfitting.
En effet, le nombre d'erreur est donc plus élevé car cela ne lui permet
pas de faire un pattern fiable et de généraliser sur les données.
C'est pourquoi à partir de 10 arbres le modèle commence à mieux généraliser, le taux d'erreur diminue donc (d'environ 12%) car le modèle devient
plus fiable. Enfin, un modèle à 100 arbres est plus précis, avec un taux d'erreur plus faible (seulement 17 %), car il peut généraliser
sur plusieurs cas avec le nombre d'abres adapté à la taille du dataset.

"""
print()
print()
print()
print("Affichage dun Decision Tree")

"""
Nous allons désormais entraîner un decision tree (et donc un seul arbre!), pour pouvoir visualiser l'arbre résultant.

Affichons désormais l'arbre de décision correspondant
(il n'est pas du tout nécessaire de lire ni de comprendre la fonction print_tree_horizontal,
il suffit juste de l'exécuter et de regarder sa sortie).

"""

#Un DT est modèle de machine learning qui prédit en posant des qs, ici si une variable cible survie
#Chaque nœud de l'arbre représente une question sur une feature ("Sex_male ≤ 0.5 ?")
#Chaque feuille donne la prédiction finale (ex: "Class=True" signifie que le passager a survécu)


decision_tree = DecisionTreeClassifier(max_depth = 3, random_state=42)
decision_tree.fit(X_train, y_train)

def print_tree_horizontal(tree, feature_names, node_id=0, indent=""):
    """
    Fonction récursive pour afficher l'arbre de décision de manière horizontale.

    Parameters:
    - tree : l'arbre à parcourir
    - feature_names : noms des caractéristiques utilisées pour les splits
    - node_id : identifiant du nœud actuel
    - indent : string qui permet de formater l'affichage des niveaux de l'arbre
    """
    # Si c'est une feuille
    if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
        values = tree.value[node_id]
        classification = 'True' if values[0][1] > values[0][0] else 'False'
        n_samples = tree.n_node_samples[node_id]
        print(f"{indent}Leaf: Class={classification}, Samples={n_samples}")
    else:
        # Si c'est un nœud de décision
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # Affiche la condition pour ce nœud
        print(f"{indent}If {feature} <= {threshold:.2f}?")

        # Branche True (gauche)
        print(f"{indent}--> True:")
        print_tree_horizontal(tree, feature_names, tree.children_left[node_id], indent + "    ")

        # Branche False (droite)
        print(f"{indent}--> False:")
        print_tree_horizontal(tree, feature_names, tree.children_right[node_id], indent + "    ")

print_tree_horizontal(decision_tree.tree_, X_encoded.columns)


print('\033[1m Quel est la feature la plus importante pour déterminer la survie ou non dun passager ?\033[0m')

"""
Dans l'exemple ci-dessus, la première question est Sex_male <= 0.5, ce qui signifie que le sexe est la feature la plus déterminante pour prédire la survie.
L'arbre de décision montre que le modèle utilise d'abord le sexe pour séparer les survivants des non-survivants. Le modèle a compris que le sexe est le meilleur critère pour
séparer les survivants des non-survivants ce qui fait écho à nos premières analyses sur le jeux de données dont les résultats ont montré que le sexe est fortement corrélé à la survie, bien plus que la feature classe.

Après le sexe, l'arbre utilise l'age. Il montre que les enfants ont plus de chances de survivre, surtout ceux ≤ 6.5 ans. Sur le Titanic, on suppose et déduit donc que les femmes et enfants ont été évacués en priorité.
Enfin, les autres features importantes sont Pclass, on observe que les passagers de 1ère classe ont plus de chances de survivre et que dans cette logique, la
la feature Fare montre que les passagers ayant payé un tarif élevé ont plus de chances de survivre. Par exemple, Les hommes en première classe (Pclass = 1) ou ayant payé un tarif élevé (Fare > 23.35) ont plus de chances de survivre.

En conclusion, les autres features comme Pclass, Age, et Fare, CAbin etc sont importantes, mais bien moins que le sexe qui est le facteur déterminant pour prédire la survie des passagers du Titanic.

"""

