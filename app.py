import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('RandomForestRegressor_model.pkl')
scaler = joblib.load('scaler.pkl')
PowerTransformers = joblib.load('PowerTransformers.pkl')

status_map = {'Developed': 0, 'Developing': 1}

image_url = "https://user-images.githubusercontent.com/86721208/164498440-5f171021-c58f-470f-863c-dbb5b0325ae4.png"

def prepare_input(p_adult_mortality, p_infant_deaths, p_alcohol, p_percentage_expenditure, p_hepatitis_b, p_measles, p_bmi, 
                p_under_five_deaths, p_polio, p_total_expenditure, p_diphtheria, p_hiv_aids, p_gdp, p_population, 
                p_thinness_1_19_years, p_thinness_5_9_years, p_income_composition_of_resources, p_schooling, p_status):
    
    adult_mortality = PowerTransformers['Adult_Mortality'].transform([[np.float64(p_adult_mortality)]])[0][0]
    infant_deaths = PowerTransformers['Infant_Deaths'].transform([[np.float64(p_infant_deaths)]])[0][0]
    alcohol = PowerTransformers['Alcohol'].transform([[p_alcohol]])[0][0]
    percentage_expenditure = PowerTransformers['Percentage_Expenditure'].transform([[p_percentage_expenditure]])[0][0]
    hepatitis_b = PowerTransformers['Hepatitis_B'].transform([[np.float64(p_hepatitis_b)]])[0][0]
    measles = PowerTransformers['Measles'].transform([[np.float64(p_measles)]])[0][0]
    under_five_deaths = PowerTransformers['Under_Five_Deaths'].transform([[np.float64(p_under_five_deaths)]])[0][0]
    polio = PowerTransformers['Polio'].transform([[np.float64(p_polio)]])[0][0]
    total_expenditure = PowerTransformers['Total_Expenditure'].transform([[p_total_expenditure]])[0][0]
    diphtheria = PowerTransformers['Diphtheria'].transform([[np.float64(p_diphtheria)]])[0][0]
    hiv_aids = PowerTransformers['Hiv/Aids'].transform([[p_hiv_aids]])[0][0]
    gdp = PowerTransformers['Gdp'].transform([[p_gdp]])[0][0]
    population = PowerTransformers['Population'].transform([[np.float64(p_population)]])[0][0]
    thinness_1_19_years = PowerTransformers['Thinness__1_19_Years'].transform([[p_thinness_1_19_years]])[0][0]
    thinness_5_9_years = PowerTransformers['Thinness_5_9_Years'].transform([[p_thinness_5_9_years]])[0][0]
    income_composition_of_resources = PowerTransformers['Income_Composition_Of_Resources'].transform([[p_income_composition_of_resources]])[0][0]
    schooling = PowerTransformers['Schooling'].transform([[np.float64(p_schooling)]])[0][0]

    input_data = np.array([[adult_mortality, infant_deaths, alcohol, percentage_expenditure, hepatitis_b, measles, p_bmi, 
                under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, population, 
                thinness_1_19_years, thinness_5_9_years, income_composition_of_resources, schooling, p_status]])
    input_data_scaled = scaler.transform(input_data)
    return input_data_scaled

def main():
    st.title("Life Expectancy Prediction", text_alignment="center")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            adult_mortality = st.number_input('Adult Mortality', min_value=1, max_value=750, value=200, step=1)
            infant_deaths = st.number_input('Infant Deaths', min_value=0, max_value=1800, value=100, step=1)
            alcohol = st.number_input('Alcohol Consumption (liters per capita)', min_value=0.01, max_value=18.0, value=5.0, step=0.01)
            percentage_expenditure = st.number_input('Percentage Expenditure (GDP per capita)', min_value=0.0, max_value=19000.0, value=300.0, step=0.01)
            hepatitis_b = st.number_input('Hepatitis B immunization coverage (%)', min_value=1, max_value=100, value=80, step=1)
            measles = st.number_input('Measles cases', min_value=0, max_value=200000, value=1000, step=1)
            bmi = st.number_input('BMI', min_value=1.0, max_value=88.0, value=30.0, step=0.1)
        with col2:
            under_five_deaths = st.number_input('Under-five Deaths', min_value=0, max_value=2500, value=100, step=1)
            polio = st.number_input('Polio immunization coverage (%)', min_value=0, max_value=100, value=80, step=1)
            total_expenditure = st.number_input('Total expenditure (% of GDP)', min_value=0.3, max_value=18.0, value=6.0, step=0.1)
            diphtheria = st.number_input('Diphtheria immunization coverage (%)', min_value=0, max_value=100, value=80, step=1)
            hiv_aids = st.number_input('HIV/AIDS (deaths per 1000 live births)', min_value=0.1, max_value=50.0, value=0.5, step=0.1)
            gdp = st.number_input('GDP per capita', min_value=1.0, max_value=120000.0, value=5000.0, step=0.1)
            population = st.number_input('Population', min_value=34, max_value=1300000000, value=1000000, step=1)
        with col3:
            thinness_1_19_years = st.number_input('Thinness 1-19 years (%)', min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            thinness_5_9_years = st.number_input('Thinness 5-9 years (%)', min_value=0.1, max_value=30.0, value=5.0, step=0.1)
            income_composition_of_resources = st.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.1, step = 0.01)
            schooling = st.number_input('Schooling (years)', min_value=0, max_value=21, value=10, step=1)
            status = st.radio('Country Status', options=list(status_map.keys()))
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict")
    if submitted:
        input_data_scaled = prepare_input(adult_mortality, infant_deaths, alcohol, percentage_expenditure, hepatitis_b, measles, bmi, 
                                          under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, population, 
                                          thinness_1_19_years, thinness_5_9_years, income_composition_of_resources, schooling, status_map[status])
        predicted_life_expectancy = model.predict(input_data_scaled)[0]
        print(f"Predicted Life Expectancy: {predicted_life_expectancy:.2f} years")
        st.success(f"Predicted Life Expectancy: {predicted_life_expectancy:.2f} years")

if __name__ == "__main__":
    main()
