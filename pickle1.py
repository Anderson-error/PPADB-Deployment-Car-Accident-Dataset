import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read the data
df = pd.read_csv('DataCleanedUp.csv')

# Pilih kolom-kolom yang akan digunakan sebagai fitur-fitur
features = [
    'Latitude', 'Longitude', 'Number_of_Casualties', 'Number_of_Vehicles', 
    'Speed_limit', 'Day_of_Week_Friday', 'Day_of_Week_Monday', 'Day_of_Week_Saturday', 
    'Day_of_Week_Sunday', 'Day_of_Week_Thursday', 'Day_of_Week_Tuesday', 'Day_of_Week_Wednesday', 
    'Light_Conditions_Darkness__lighting_unknown', 'Light_Conditions_Darkness__lights_lit', 
    'Light_Conditions_Darkness__lights_unlit', 'Light_Conditions_Darkness__no_lighting', 
    'Light_Conditions_Daylight', 'Road_Type_Dual_carriageway', 'Road_Type_One_way_street', 
    'Road_Type_Roundabout', 'Road_Type_Single_carriageway', 'Road_Type_Slip_road', 
    'Urban_or_Rural_Area_Rural', 'Urban_or_Rural_Area_Urban', 
    'Weather_Conditions_Fine__high_winds', 'Weather_Conditions_Fine_no_high_winds', 
    'Weather_Conditions_Fog_or_mist', 'Weather_Conditions_Other', 
    'Weather_Conditions_Raining__high_winds', 'Weather_Conditions_Raining_no_high_winds', 
    'Weather_Conditions_Snowing__high_winds', 'Weather_Conditions_Snowing_no_high_winds'
]

# Split the data into features and target variable
X = df[features]  # Features
y = df['Accident_Severity']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('CarAccidentModel.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'CarAccidentModel.pkl'")