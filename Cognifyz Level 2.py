import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
import folium

file_path = r"C:\Users\HP\.vscode\Task\Dataset .csv"
data = pd.read_csv(file_path)

# Task 1: Restaurant Ratings

plt.figure(figsize=(10, 6))
sns.histplot(data['Aggregate rating'], bins=20, kde=True)
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()

common_rating_range = data['Aggregate rating'].value_counts().idxmax()
print(f"The most common rating is: {common_rating_range}")

average_votes = data['Votes'].astype(int).mean()
print(f"The average number of votes received by restaurants is: {average_votes:.2f}")

# Task 2: Cuisine Combination

cuisines = data['Cuisines'].dropna().str.split(', ')
cuisine_combinations = Counter([tuple(sorted(combo)) for sublist in cuisines for combo in combinations(sublist, 2)])
most_common_combinations = cuisine_combinations.most_common(10)
print("The most common cuisine combinations are:")
for combo, count in most_common_combinations:
    print(f"{combo}: {count}")

cuisine_rating = {}
for combo, count in cuisine_combinations.items():
    mask = data['Cuisines'].apply(lambda x: all(c in x for c in combo) if isinstance(x, str) else False)
    avg_rating = data[mask]['Aggregate rating'].mean()
    cuisine_rating[combo] = avg_rating
sorted_cuisine_rating = sorted(cuisine_rating.items(), key=lambda x: x[1], reverse=True)
print("Cuisine combinations with the highest average ratings:")
for combo, avg_rating in sorted_cuisine_rating[:10]:
    print(f"{combo}: {avg_rating:.2f}")

# Task 3: Geographic Analysis

map_center = [data['Latitude'].median(), data['Longitude'].median()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)
for idx, row in data.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(restaurant_map)
restaurant_map.save("restaurant_map.html")
print("Map has been saved as restaurant_map.html")

print("Inspect the saved map to identify clusters.")

# Task 4: Restaurant Chains

restaurant_chains = data['Restaurant Name'].value_counts()
chains = restaurant_chains[restaurant_chains > 1]
print("Restaurant chains identified in the dataset:")
print(chains)

chain_analysis = data[data['Restaurant Name'].isin(chains.index)].groupby('Restaurant Name').agg({
    'Aggregate rating': 'mean',
    'Votes': 'sum'
}).sort_values(by='Aggregate rating', ascending=False)
print("Ratings and popularity of different restaurant chains:")
print(chain_analysis)
