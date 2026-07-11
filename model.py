import pandas as pd

# Step 1: Load the Kaggle dataset
train_df = pd.read_csv('data/Training.csv')

# Step 2: View basic info
print("Original dataset shape:", train_df.shape)
print("Number of unique diseases:", train_df['prognosis'].nunique())

# Step 3: Count disease frequency
disease_counts = train_df['prognosis'].value_counts().head(20)
top_20_diseases = disease_counts.index.tolist()
print("Top 20 diseases selected:\n", top_20_diseases)

# Step 4: Filter rows with only top 20 diseases
filtered_df = train_df[train_df['prognosis'].isin(top_20_diseases)]

# Step 5: Keep only numeric columns (symptoms)
symptom_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()

# Step 6: Pick 20 most varying symptoms
symptom_variance = filtered_df[symptom_columns].var().sort_values(ascending=False)
top_20_symptoms = symptom_variance.head(20).index.tolist()

# Step 7: Keep these symptoms + disease
final_df = filtered_df[top_20_symptoms + ['prognosis']]

# Step 8: Save as a simpler dataset
final_df.to_csv('data/symptoms_disease.csv', index=False)

print("Filtered dataset saved as 'symptoms_disease.csv'")
print("Shape:", final_df.shape)


# -----------------------------------------------------------
# STEP 2: Train Naive Bayes Model on the Filtered Dataset
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load the filtered dataset
df = pd.read_csv('data/symptoms_disease.csv')

# Split features (X) and target (y)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model trained successfully with accuracy: {acc * 100:.2f}%")

# Save model and feature columns for Flask app use later
joblib.dump(model, 'data/disease_model.pkl')
joblib.dump(X.columns.tolist(), 'data/symptom_columns.pkl')

print("Model and symptom columns saved in 'data/' folder.")

