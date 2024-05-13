import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import locale
from datetime import datetime

# Chargement des données
data = pd.read_csv('atomic_data.csv')

# Conversion de la colonne 'Transaction Date' en datetime
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])

data['CA Vente'] = data['Quantity'] * data['Unit Price']

# Objectif 1: Chiffre d'affaires total
total_revenue = data['CA Vente'].sum()
total_revenue_arrond = round(total_revenue, 2)

# Objectif 2: Classement des Produits par Chiffre d'Affaires
product_revenue = data.groupby('Product Name')['CA Vente'].sum().sort_values(ascending=False)

# Objectif 3: Moyen de Paiement le Plus Utilisé
payment_method_counts = data['Payment Method'].value_counts()
most_used_payment_method = payment_method_counts.idxmax()

# Objectif 4: Pays avec les Ventes les Plus Élevées
country_revenue = data.groupby('Country')['CA Vente'].sum()
top_country = country_revenue.idxmax()

# Objectif 5: Tendance des Ventes en Fonction du Temps (regroupé par mois)
monthly_revenue = data.resample('MS', on='Transaction Date')['CA Vente'].sum()

# Supprimer les doublons de mois pour l'affichage
monthly_revenue = monthly_revenue.groupby(monthly_revenue.index.month).sum()

# Objectif 6: Prédiction du Chiffre d'Affaires pour Mai 2024
model = ARIMA(monthly_revenue, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=1)
forecast_rounded = round(forecast, 2)

# Interface Streamlit
st.header('BATCHO')
st.title('Analyse des Ventes en Ligne')

# Objectif 1: Chiffre d'affaires total
st.header('Chiffre d\'affaires total')
st.write(f"Le chiffre d'affaires total est : {total_revenue_arrond}")

# Objectif 2: Classement des Produits par Chiffre d'Affaires
st.header('Classement des Produits par Chiffre d\'Affaires')
st.write(product_revenue)

# Objectif 3: Moyen de Paiement le Plus Utilisé
st.header('Moyen de Paiement le Plus Utilisé')
st.write(f"Le moyen de paiement le plus utilisé est : {most_used_payment_method}")

# Objectif 4: Pays avec les Ventes les Plus Élevées
st.header('Pays avec les Ventes les Plus Élevées')
st.write(f"Le pays avec les ventes les plus élevées est : {top_country}")

# Objectif 5: Tendance des Ventes en Fonction du Temps (regroupé par mois)
st.header('Tendance des Ventes en Fonction du Temps (par mois)')
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
monthly_revenue = data.resample('MS', on='Transaction Date')['CA Vente'].sum()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_revenue.index, monthly_revenue.values, label='Chiffre d\'affaires mensuel', color='blue')
ax.set_xlabel('Mois')
ax.set_ylabel('Chiffre d\'affaires mensuel')
ax.set_title('Tendance des ventes mensuelles')
#ax.tick_params(axis='x', rotation=45,)  # Rotation de l'étiquette de l'axe des abscisses
plt.xticks(monthly_revenue.index, [month.strftime('%B %Y') for month in monthly_revenue.index], rotation=45)
ax.legend()
st.pyplot(fig)

# Objectif 6: Prédiction du Chiffre d'Affaires pour Mai 2024
st.header('Prédiction du Chiffre d\'Affaires pour Mai 2024')
st.write(f"Prévision du chiffre d'affaires pour Mai 2024 : {forecast_rounded}")
