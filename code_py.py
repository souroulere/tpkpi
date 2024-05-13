import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Chargement des données
data = pd.read_csv('atomic_data.csv')

# Conversion de la colonne 'Transaction Date' en datetime
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])

data['CA Vente']=data['Quantity'] * data['Unit Price']
print(data.head())

# Objectif 1: Chiffre d'affaires total
total_revenue = data['Quantity'] * data['Unit Price']
print("Chiffre d'affaires total:", total_revenue.sum())

# Objectif 2: Classement des Produits par Chiffre d'Affaires
product_revenue = data.groupby('Product Name')['Quantity'].sum() * data.groupby('Product Name')['Unit Price'].mean()
sorted_products = product_revenue.sort_values(ascending=False)
print("Classement des produits par chiffre d'affaires:")
print(sorted_products)

# Objectif 3: Moyen de Paiement le Plus Utilisé
payment_method_counts = data['Payment Method'].value_counts()
print("Moyen de paiement le plus utilisé:", payment_method_counts.idxmax())

# Objectif 4: Pays avec les Ventes les Plus Élevées
#country_revenue = data.groupby('Country')['Quantity'].sum() * data.groupby('Country')['Unit Price'].mean()
country_revenue = data.groupby('Country')['CA Vente'].sum()
top_country = country_revenue.idxmax()
print("Pays avec les ventes les plus élevées:", top_country)

# Objectif 5: Tendance des Ventes en Fonction du Temps
#monthly_revenue = data.resample('ME', on='Transaction Date')['Quantity'].sum()
monthly_revenue = data.resample('ME', on='Transaction Date')['CA Vente'].sum()

print(monthly_revenue)
"""
plt.plot(monthly_revenue.index, monthly_revenue.values)
plt.xlabel('Mois')
plt.ylabel('Chiffre d\'affaires mensuel')
plt.title('Tendance des ventes mensuelles')
plt.xticks(rotation='vertical')
"""
# Créer une figure et des sous-graphiques
fig, ax = plt.subplots()

# Créer l'histogramme
ax.bar(monthly_revenue.index, monthly_revenue.values, label='Chiffre d\'affaires mensuel')

# Tracer la courbe
ax.plot(monthly_revenue.index, monthly_revenue.values, color='red', label='Tendance')

# Ajouter des étiquettes et un titre
ax.set_xlabel('Mois')
ax.set_ylabel('Chiffre d\'affaires mensuel')
ax.set_title('Tendance des ventes mensuelles')

# Spécifier l'orientation du texte sur l'axe des abscisses
plt.xticks(rotation='vertical')

# Afficher la légende
ax.legend()
plt.show()

# Objectif 6: Prédiction du Chiffre d'Affaires pour Mai 2024 (Exemple)
# Utilisez des techniques de prévision pour prédire le chiffre d'affaires futur


# Chargement des données (assurez-vous que vos données sont prêtes)
# Exemple : data est votre DataFrame avec le chiffre d'affaires mensuel
# On utilise monthly_revenue

# Création du modèle ARIMA
model = ARIMA(monthly_revenue, order=(1, 1, 1))  # Exemple d'ordre ARIMA, à ajuster selon vos données

# Entraînement du modèle
model_fit = model.fit()

# Prévision pour Mai 2024 (par exemple, en supposant que votre index est une date)
forecast = model_fit.forecast(steps=1)  # Prévision pour 1 pas (1 mois dans ce cas)

# Affichage de la prévision
print("Prévision du chiffre d'affaires pour Mai 2024 :", forecast)

# Visualisation des prévisions
plt.plot(monthly_revenue.index, monthly_revenue.values, label='Historique')
plt.plot(pd.to_datetime(['2024-05']), forecast, 'r--', label='Prévision Mai 2024')
plt.xlabel('Date')
plt.ylabel('Chiffre d\'affaires')
plt.title('Prévision du Chiffre d\'Affaires')
plt.legend()
plt.show()

