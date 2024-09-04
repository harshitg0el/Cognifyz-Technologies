import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Dataset .csv")

# Convert 'Yes'/'No' columns to numeric
data['Has Online delivery'] = data['Has Online delivery'].map({'Yes': 1, 'No': 0})
data['Has Table booking'] = data['Has Table booking'].map({'Yes': 1, 'No': 0})

# Print column names to ensure we have the correct columns
print("Columns in the dataset:", data.columns)

# Task 1: Restaurant Reviews Analysis
# Assuming the text you want to analyze is in the 'Rating text' or another similar column

def extract_keywords(text_series):
    vectorizer = CountVectorizer(stop_words='english')
    word_count = vectorizer.fit_transform(text_series.dropna())
    word_freq = pd.DataFrame(word_count.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)
    return word_freq

# Since there's no direct 'reviews' column, using 'Rating text'
positive_reviews = data[data['Aggregate rating'] >= 4]['Rating text']
negative_reviews = data[data['Aggregate rating'] <= 2]['Rating text']

positive_keywords = extract_keywords(positive_reviews)
negative_keywords = extract_keywords(negative_reviews)

print("Top Positive Keywords:", positive_keywords.head(10))
print("Top Negative Keywords:", negative_keywords.head(10))

# 1.2 Average Length of Reviews and Relationship with Rating
# Assuming 'Rating text' is the review, or you could replace this with the correct column

data['review_length'] = data['Rating text'].apply(lambda x: len(str(x).split()))
average_review_length = data['review_length'].mean()

sns.scatterplot(data=data, x='review_length', y='Aggregate rating')
plt.title('Review Length vs Rating')
plt.show()

print(f"Average review length: {average_review_length}")

# Task 2: Votes Analysis
# 2.1 Identify Restaurants with the Highest and Lowest Number of Votes
highest_votes = data[data['Votes'] == data['Votes'].max()]
lowest_votes = data[data['Votes'] == data['Votes'].min()]

print("Restaurant with the highest votes:", highest_votes[['Restaurant Name', 'Votes']])
print("Restaurant with the lowest votes:", lowest_votes[['Restaurant Name', 'Votes']])

# 2.2 Correlation Between Number of Votes and Rating
correlation_votes_rating = data['Votes'].corr(data['Aggregate rating'])
print(f"Correlation between votes and rating: {correlation_votes_rating}")

sns.scatterplot(data=data, x='Votes', y='Aggregate rating')
plt.title('Votes vs Rating')
plt.show()

# Task 3: Price Range vs. Online Delivery and Table Booking
# 3.1 Relationship Between Price Range and Availability of Online Delivery and Table Booking
delivery_booking_relation = data.groupby('Price range').agg({
    'Has Online delivery': 'mean',
    'Has Table booking': 'mean'
}).reset_index()

sns.barplot(data=delivery_booking_relation, x='Price range', y='Has Online delivery')
plt.title('Price Range vs Online Delivery')
plt.show()

sns.barplot(data=delivery_booking_relation, x='Price range', y='Has Table booking')
plt.title('Price Range vs Table Booking')
plt.show()

# 3.2 Higher Priced Restaurants Offering Services
higher_priced_services = data.groupby('Price range').agg({
    'Has Online delivery': 'mean',
    'Has Table booking': 'mean'
}).sort_values('Price range', ascending=False)

print(higher_priced_services)
