import streamlit as st
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"E:\da\mini\nwfl1.csv")

# Extract company from name and drop name column
df['Company'] = df['name'].apply(lambda x: x.split()[0])
df.drop(columns=['name'], inplace=True)

# Replace categorical variables with numerical values
categorical_mapping = {
    "fuel": {"Diesel": 1, "Petrol": 2, "LPG": 3, "CNG": 4},
    "seller_type": {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3},
    "transmission": {'Manual': 1, "Automatic": 2},
    "owner": {'First Owner': 1, "Second Owner": 2, "Third Owner": 3, "Fourth & Above Owner": 4, "Test Drive Car": 5},
    "Company": {'Skoda': 1, 'Honda': 2, 'Hyundai': 3, 'Maruti': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
                'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Fiat': 11, 'Datsun': 12, 'Jeep': 13,
                'Mercedes-Benz': 14, 'Mitsubishi': 15, 'Audi': 16, 'Volkswagen': 17, 'BMW': 18,
                'Nissan': 19, 'Lexus': 20, 'Jaguar': 21, 'Land': 22, 'MG': 23, 'Volvo': 24, 'Daewoo': 25,
                'Kia': 26, 'Force': 27, 'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31, 'Peugeot': 32}
}

df.replace(categorical_mapping, inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Train the Decision Tree Regressor
X = df_imputed.drop('selling_price', axis=1)
y = df_imputed['selling_price']
regressor = DecisionTreeRegressor()
regressor.fit(X, y)
reverse_company_mapping = {v: k for k, v in categorical_mapping['Company'].items()}

# User inputs
st.title("USED CAR PRICE PREDICTION")
st.sidebar.header("User Inputs")

# Sidebar inputs
year = st.sidebar.slider("Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()))
km_driven = st.sidebar.slider("Kilometers driven", min_value=0, max_value=int(df['km_driven'].max()))
fuel = st.sidebar.selectbox("Fuel type", options=["Diesel", "Petrol", "LPG", "CNG"])
seller_type = st.sidebar.radio("Seller type", options=["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.radio("Transmission", options=['Manual', 'Automatic'])
owner = st.sidebar.radio("Owner", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.sidebar.slider("Mileage", min_value=0.0, max_value=float(df['mileage'].max()))
engine = st.sidebar.slider("Engine capacity", min_value=0.0, max_value=float(df['engine'].max()))
seats = st.sidebar.slider("Number of seats", min_value=0, max_value=int(df['seats'].max()))
company = st.selectbox("Company", options=list(reverse_company_mapping.values()))

# Convert categorical inputs to numerical
fuel = categorical_mapping['fuel'][fuel]
seller_type = categorical_mapping['seller_type'][seller_type]
transmission = categorical_mapping['transmission'][transmission]
owner = categorical_mapping['owner'][owner]

# Predict price
if st.button("Predict"):
    feature_values = [[year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, seats, categorical_mapping['Company'][company]]]
    prediction = regressor.predict(feature_values)[0]
    st.success(f"Predicted price is: {prediction:.2f} INR")

# Clustering and Visualization for selected data input by the user
st.title("CLASSIFICATION CLUSTERING")
if st.button("analyze"):
    st.title("Classification Task")
    target_variable = 'seller_type'
    X_classification = df_imputed.drop(columns=target_variable)
    y_classification = df_imputed[target_variable]

    # Train the Decision Tree Classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_classification, y_classification)

    # Evaluate model
    accuracy = classifier.score(X_classification, y_classification)
    st.write(f"Accuracy: {accuracy:.2f}")

    # Clustering for selected data input by the user
    st.title("Clustering Task")
    selected_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'seats': [seats],
        'Company': [categorical_mapping['Company'][company]]
    })

    # Prepare data for clustering
    features_for_clustering = ['mileage', 'engine', 'seats']
    X_clustering = df_imputed[features_for_clustering]

    # Add the selected data to the clustering dataset
    X_clustering = pd.concat([X_clustering, selected_data[features_for_clustering]], ignore_index=True)

    # Scale data
    scaler = StandardScaler()
    X_clustering_scaled = scaler.fit_transform(X_clustering)

    # Apply KMeans clustering
    k = 3  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(X_clustering_scaled)

    # Add cluster labels to the dataframe
    df_imputed['Cluster'] = clusters[:-1]  # All data except the last row (selected data)
    selected_data['Cluster'] = clusters[-1]  # Last row is the selected data

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_imputed, x='mileage', y='engine', hue='Cluster', palette='viridis', ax=ax)
    sns.scatterplot(data=selected_data, x='mileage', y='engine', hue='Cluster', color='red', marker='X', s=200, ax=ax)
    ax.set_title('Clustering of Cars')
    st.pyplot(fig)

# Graphical Visualization
st.title("GRAPHICAL VISUALIZATION")
if st.button("visualize"):
    st.subheader("Visualizations")

    st.write("### Box Plot of Selling Price by Fuel Type")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df_imputed, x='fuel', y='selling_price', ax=ax)
    ax.set_title("Selling Price by Fuel Type")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    st.write("### Scatter Plot of Mileage vs. Engine")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_imputed, x='mileage', y='engine', hue='selling_price', palette='viridis', ax=ax)
    ax.set_title("Mileage vs. Engine Capacity")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Engine Capacity")
    st.pyplot(fig)

    st.write("### Bar Plot of Number of Cars by Transmission Type")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=df_imputed, x='transmission', palette='viridis', ax=ax)
    ax.set_title("Number of Cars by Transmission Type")
    ax.set_xlabel("Transmission Type")
    ax.set_ylabel("Number of Cars")
    st.pyplot(fig)

    st.subheader("Company Specific Data")
    company_data = df_imputed[df_imputed['Company'] == categorical_mapping['Company'][company]]

    # Number of cars in the selected year for the selected company
    year_data = company_data[company_data['year'] == year]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=year_data, x='year', palette='viridis', ax=ax)
    ax.set_title(f"Number of Cars for {company} in Year {year}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cars")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(data=company_data, x='year', palette='viridis', ax=ax)
    ax.set_title(f"Number of Cars for {company} Over the Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cars")
    st.pyplot(fig)

    st.write("### Box Plot of Selling Price by Year for Selected Company")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=company_data, x='year', y='selling_price', ax=ax)
    ax.set_title(f"Selling Price by Year for {company}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    st.write("### Scatter Plot of Mileage vs. Selling Price for Selected Company")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=company_data, x='mileage', y='selling_price', ax=ax)
    ax.set_title(f"Mileage vs. Selling Price for {company}")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)

    st.write("### Engine Capacity vs. Selling Price for Selected Company")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=company_data, x='engine', y='selling_price', ax=ax)
    ax.set_title(f"Engine Capacity vs. Selling Price for {company}")
    ax.set_xlabel("Engine Capacity")
    ax.set_ylabel("Selling Price")
    st.pyplot(fig)
