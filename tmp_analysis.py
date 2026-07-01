import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unicodedata import normalize


def normalize_country(name):
    if pd.isna(name):
        return None
    name = str(name).strip().lower()
    name = normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = name.replace("'", "").replace('.', '').replace(' ', '')
    return name

pib = pd.read_csv('macrodata/gdp-per-capita-worldbank.csv')
players = pd.read_csv('football/players.csv')

pib['country_name_norm'] = pib['Entity'].apply(normalize_country)
players['country_of_birth_norm'] = players['country_of_birth'].apply(normalize_country)

pib_2015 = pib[pib['Year'] == 2015].copy()

# Elite definition: top 10% by market value within each country
player_value = players[['country_of_birth_norm', 'market_value_in_eur']].dropna(subset=['country_of_birth_norm', 'market_value_in_eur'])
threshold = player_value.groupby('country_of_birth_norm')['market_value_in_eur'].quantile(0.90).rename('elite_threshold')
player_value = player_value.merge(threshold, on='country_of_birth_norm', how='left')
player_value['is_elite'] = player_value['market_value_in_eur'] >= player_value['elite_threshold']

country_stats = (
    player_value.groupby('country_of_birth_norm', as_index=False)
    .agg(
        n_jogadores=('market_value_in_eur', 'size'),
        avg_market_value=('market_value_in_eur', 'mean'),
        n_elite=('is_elite', 'sum')
    )
)

country_stats = country_stats.merge(
    pib_2015[['country_name_norm', 'GDP per capita']],
    left_on='country_of_birth_norm',
    right_on='country_name_norm',
    how='left'
)

country_stats = country_stats.dropna(subset=['GDP per capita', 'avg_market_value'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=country_stats, x='GDP per capita', y='avg_market_value', size='n_jogadores', sizes=(80, 300), alpha=0.8)
for _, row in country_stats.iterrows():
    plt.text(row['GDP per capita'] + 0.01 * max(country_stats['GDP per capita']), row['avg_market_value'] + 0.01 * max(country_stats['avg_market_value']), row['country_of_birth_norm'], fontsize=8)
plt.title('PIB x valor médio de mercado dos jogadores por país')
plt.xlabel('PIB per capita')
plt.ylabel('Valor médio de mercado')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=country_stats, x='GDP per capita', y='n_elite', size='n_jogadores', sizes=(80, 300), alpha=0.8)
for _, row in country_stats.iterrows():
    plt.text(row['GDP per capita'] + 0.01 * max(country_stats['GDP per capita']), row['n_elite'] + 0.1, row['country_of_birth_norm'], fontsize=8)
plt.title('PIB x quantidade de jogadores de elite por país')
plt.xlabel('PIB per capita')
plt.ylabel('Quantidade de jogadores de elite')
plt.tight_layout()
plt.show()

print(country_stats.head())
