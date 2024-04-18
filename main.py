import pickle
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('DataCleanedUp.csv')
df.columns = [col.replace("-", "").replace("+", "").replace(" ", "_") for col in df.columns]

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


with open('CarAccidentModel.pkl', 'rb') as f:
        car_model = pickle.load(f)

def main():
    st.set_page_config(page_title="Car Accident Dataset", page_icon="📊")
    # Sidebar
    st.sidebar.title("Pages")
    app_mode = st.sidebar.selectbox("Choose your pages :",
        ["Home", "Data", "Modeling"])
    
    if app_mode == "Home":
        
        st.title("Car Accident Dataset")
        st.image("https://storage.googleapis.com/kaggle-datasets-images/4288589/7379695/e7312ba90e4ca28045f6d450b1fcfc9d/dataset-cover.png?t=2024-01-10-23-05-47", use_column_width=True)
        st.header("BUSINESS UNDERSTANDING")
        st.header("Bussiness Objective")
        st.write('''Tujuan bisnis dari dataset ini adalah untuk secara menyeluruh menggali dan memahami faktor-faktor yang berperan dalam terjadinya kecelakaan jalan di daerah perkotaan selama tahun tersebut.
                 Melalui analisis yang mendalam terhadap data kecelakaan yang terinci, dengan tujuan untuk menyediakan wawasan yang komprehensif kepada pemangku kepentingan terkait tentang faktor-faktor seperti tanggal kejadian, 
                 hari dalam seminggu, kendali persimpangan, tingkat keparahan kecelakaan, kondisi geografis, kondisi pencahayaan dan cuaca, serta detail kendaraan yang terlibat. 
                 Dengan pemahaman yang lebih baik tentang dinamika kecelakaan jalan yang terjadi selama tahun 2021, kami bertujuan untuk mendukung pengembangan strategi keselamatan jalan yang lebih efektif. 
                 Langkah-langkah yang diambil dari analisis ini diharapkan dapat mengarah pada inisiatif pencegahan yang lebih efisien dan program keselamatan jalan yang lebih cermat, dengan tujuan akhir mengurangi jumlah 
                 insiden kecelakaan di masa depan dan meningkatkan keselamatan bagi semua pengguna jalan.''')
        
        st.header("Assess Situation")
        st.write('''Situasi bisnis yang mendasari dari analisis ini adalah Kurangnya Kepedulian Pengemudi akan Keselamatan Pengemudi di wilayah perkotaan menunjukkan tingkat kesadaran yang rendah akan pentingnya keselamatan di jalan raya, 
                 kemudian Kondisi Jalan yang Minim akan Pencahayaan. Beberapa ruas jalan di wilayah perkotaan memiliki pencahayaan yang minim terutama pada malam hari Kondisi ini dapat menciptakan lingkungan yang kurang aman bagi pengguna jalan dan meningkatkan 
                 risiko kecelakaan terutama dalam kondisi cuaca buruk atau visibilitas rendah dan Kondisi Cuaca yang Ekstrem di Wilayah perkotaan rentan terhadap kondisi cuaca ekstrem, seperti hujan deras, kabut tebal, atau salju. 
                 Kondisi cuaca yang buruk dapat menyebabkan jalan menjadi licin dan mempengaruhi visibilitas.''')
        
        st.header("Data Mining Goals")
        st.write('''Tujuan utama dari data mining adalah untuk mengidentifikasi pola-pola yang tidak terlihat dan hubungan yang signifikan dalam dataset kecelakaan. Data mining bertujuan untuk menggali informasi yang dapat membantu dalam 
                 memahami akar penyebab dari kecelakaan tersebut. Ini mencakup pencarian pola-pola seperti korelasi antara kondisi cuaca tertentu dengan tingkat kecelakaan, pola perilaku pengemudi yang berkontribusi terhadap insiden, atau kecenderungan 
                 tertentu dalam lokasi atau waktu kejadian kecelakaan.''')
        
        st.header("Project Plan")
        st.write('''Proyek ini akan dimulai dengan pengumpulan dan pembersihan data kecelakaan jalan tahun 2021 di wilayah perkotaan. Selanjutnya, akan dilakukan analisis data mining untuk mengidentifikasi pola-pola dan hubungan-hubungan antara variabel-variabel dalam dataset. 
                 Hasil analisis akan disajikan dalam laporan yang berisi temuan-temuan utama dan rekomendasi untuk langkah-langkah perbaikan keselamatan jalan. Rekomendasi akan dievaluasi dan diimplementasikan oleh pihak berwenang, 
                 dengan pemantauan terhadap dampaknya terhadap tingkat kecelakaan secara keseluruhan.''')
    
    elif app_mode == "Data":
        st.title("Data Frame")

        # Baca data dari file CSV
        chart_data = pd.read_csv("DataCleanedUp.csv")
        chart_data.columns = [col.replace("-", "").replace("+", "").replace(" ", "_") for col in chart_data.columns]
        # discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
        # df_discretized = pd.DataFrame(discretizer.fit_transform(chart_data), columns=chart_data.columns)

        st.title("Exploratory Data Analysis (EDA)")
        st.write("Performing EDA on the dataset...")
        st.write("### Dataset Overview")
        st.write(chart_data)
        st.write("Nama-nama Kolom:")
        for column, dtype in chart_data.dtypes.items():
            st.write(f"Column: {column}, Dtype: {dtype}")

        st.write("### Summary Statistics")
        st.write(chart_data.describe())

        st.write("### Distribution of Numerical Features")
        plt.figure(figsize=(10, 6))
        chart_data.select_dtypes(include=['float64', 'int64']).hist(bins=20, color='skyblue', edgecolor='black', linewidth=0.5)
        plt.xticks(fontsize=10)  # Atur ukuran teks sumbu x menjadi 10
        plt.yticks(fontsize=10)  # Atur ukuran teks sumbu y menjadi 10
        plt.tight_layout()
        st.pyplot()

        st.write("### Correlation Matrix")
        plt.figure(figsize=(50, 48))
        sns.heatmap(chart_data.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot()

        numerical_columns = chart_data.select_dtypes(include=['float64']).columns
        st.write("### Distribution of Numerical Features")
        for column in numerical_columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=chart_data[column], kde=True, color='skyblue')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot()

    elif app_mode == "Modeling":
        st.header("Association Rule Modeling")
        st.write("Data Mining Car Accident")

        # with open('CarAccidentModel.pkl', 'rb') as f:
        #     car_model = pickle.load(f)

        # Accident_Severity = st.text_input('Input nilai Accident_Severity')
        Latitude = st.text_input('Input nilai Latitude')
        Longitude = st.text_input('Input nilai Longitude')
        Number_of_Casualties = st.text_input('Input nilai Number_of_Casualties')
        Number_of_Vehicles = st.text_input('Input nilai Number_of_Vehicles')
        Speed_limit = st.text_input('Input nilai Speed_limit')
        Day_of_Week_Friday = st.text_input('Input nilai Day_of_Week_Friday')
        Day_of_Week_Monday = st.text_input('Input nilai Day_of_Week_Monday')
        Day_of_Week_Saturday = st.text_input('Input nilai Day_of_Week_Saturday')
        Day_of_Week_Sunday = st.text_input('Input nilai Day_of_Week_Sunday')
        Day_of_Week_Thursday = st.text_input('Input nilai Day_of_Week_Thursday')
        Day_of_Week_Tuesday = st.text_input('Input nilai Day_of_Week_Tuesday')
        Day_of_Week_Wednesday = st.text_input('Input nilai Day_of_Week_Wednesday')
        Light_Conditions_Darkness__lighting_unknown = st.text_input('Input nilai Light_Conditions_Darkness__lighting_unknown')
        Light_Conditions_Darkness__lights_lit = st.text_input('Input nilai Light_Conditions_Darkness__lights_lit')
        Light_Conditions_Darkness__lights_unlit = st.text_input('Input nilai Light_Conditions_Darkness__lights_unlit')
        Light_Conditions_Darkness__no_lighting = st.text_input('Input nilai Light_Conditions_Darkness__no_lighting')
        Light_Conditions_Daylight = st.text_input('Input nilai Light_Conditions_Daylight')
        Road_Type_Dual_carriageway = st.text_input('Input nilai Road_Type_Dual_carriageway')
        Road_Type_One_way_street = st.text_input('Input nilai Road_Type_One_way_street')
        Road_Type_Roundabout = st.text_input('Input nilai Road_Type_Roundabout')
        Road_Type_Single_carriageway = st.text_input('Input nilai Road_Type_Single_carriageway')
        Road_Type_Slip_road = st.text_input('Input nilai Road_Type_Slip_road')
        Urban_or_Rural_Area_Rural = st.text_input('Input nilai Urban_or_Rural_Area_Rural')
        Urban_or_Rural_Area_Urban = st.text_input('Input nilai Urban_or_Rural_Area_Urban')
        Weather_Conditions_Fine__high_winds = st.text_input('Input nilai Weather_Conditions_Fine__high_winds')
        Weather_Conditions_Fine_no_high_winds = st.text_input('Input nilai Weather_Conditions_Fine_no_high_winds')
        Weather_Conditions_Fog_or_mist = st.text_input('Input nilai Weather_Conditions_Fog_or_mist')
        Weather_Conditions_Other = st.text_input('Input nilai Weather_Conditions_Other')
        Weather_Conditions_Raining__high_winds = st.text_input('Input nilai Weather_Conditions_Raining__high_winds')
        Weather_Conditions_Raining_no_high_winds = st.text_input('Input nilai Weather_Conditions_Raining_no_high_winds')
        Weather_Conditions_Snowing__high_winds = st.text_input('Input nilai Weather_Conditions_Snowing__high_winds')
        Weather_Conditions_Snowing_no_high_winds = st.text_input('Input nilai Weather_Conditions_Snowing_no_high_winds')

        # Check if all input values are provided
        if st.button('Predict'):
            if (Latitude is not None and Longitude is not None and Number_of_Casualties is not None 
                and Number_of_Vehicles is not None and Speed_limit is not None 
                and Day_of_Week_Friday is not None and Day_of_Week_Monday is not None 
                and Day_of_Week_Saturday is not None and Day_of_Week_Sunday is not None 
                and Day_of_Week_Thursday is not None and Day_of_Week_Tuesday is not None 
                and Day_of_Week_Wednesday is not None and Light_Conditions_Darkness__lighting_unknown is not None 
                and Light_Conditions_Darkness__lights_lit is not None 
                and Light_Conditions_Darkness__lights_unlit is not None 
                and Light_Conditions_Darkness__no_lighting is not None 
                and Light_Conditions_Daylight is not None and Road_Type_Dual_carriageway is not None 
                and Road_Type_One_way_street is not None and Road_Type_Roundabout is not None 
                and Road_Type_Single_carriageway is not None and Road_Type_Slip_road is not None 
                and Urban_or_Rural_Area_Rural is not None and Urban_or_Rural_Area_Urban is not None 
                and Weather_Conditions_Fine__high_winds is not None and Weather_Conditions_Fine_no_high_winds is not None 
                and Weather_Conditions_Fog_or_mist is not None and Weather_Conditions_Other is not None 
                and Weather_Conditions_Raining__high_winds is not None and Weather_Conditions_Raining_no_high_winds is not None 
                and Weather_Conditions_Snowing__high_winds is not None and Weather_Conditions_Snowing_no_high_winds is not None):
                        
                # Convert Latitude and Longitude to float
                try:
                    Latitude = float(Latitude)
                    Longitude = float(Longitude)
                except ValueError:
                    st.error("Nilai Latitude dan Longitude harus berupa angka.")
                    return

                # Prepare input for prediction
                input_data = [[Latitude, Longitude, Number_of_Casualties, Number_of_Vehicles, 
                            Speed_limit, Day_of_Week_Friday, Day_of_Week_Monday, Day_of_Week_Saturday, 
                            Day_of_Week_Sunday, Day_of_Week_Thursday, Day_of_Week_Tuesday, Day_of_Week_Wednesday, 
                            Light_Conditions_Darkness__lighting_unknown, Light_Conditions_Darkness__lights_lit, 
                            Light_Conditions_Darkness__lights_unlit, Light_Conditions_Darkness__no_lighting, 
                            Light_Conditions_Daylight, Road_Type_Dual_carriageway, Road_Type_One_way_street, 
                            Road_Type_Roundabout, Road_Type_Single_carriageway, Road_Type_Slip_road, 
                            Urban_or_Rural_Area_Rural, Urban_or_Rural_Area_Urban, 
                            Weather_Conditions_Fine__high_winds, Weather_Conditions_Fine_no_high_winds, 
                            Weather_Conditions_Fog_or_mist, Weather_Conditions_Other, 
                            Weather_Conditions_Raining__high_winds, Weather_Conditions_Raining_no_high_winds, 
                            Weather_Conditions_Snowing__high_winds, Weather_Conditions_Snowing_no_high_winds]]

                # Perform prediction
                car_accident = car_model.predict(input_data)

                # Map prediction to "accident" or "no accident"
                prediction_result = "There was an accident incident" if car_accident[0] == 1 else "There were no accident incidents"

                # Display prediction
                st.write(f'Predicted accident: {prediction_result}')
            else:
                st.error("Mohon isi semua nilai input sebelum melakukan prediksi.")



if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)
    main()