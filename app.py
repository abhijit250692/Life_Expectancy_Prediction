import streamlit as st
import joblib
import numpy as np

# Load the best model
model = joblib.load('RandomForestRegressor_model.pkl')
scaler = joblib.load('scaler.pkl')
status_map = {'Developed': 0, 'Developing': 1}
image_url = "https://user-images.githubusercontent.com/86721208/164498440-5f171021-c58f-470f-863c-dbb5b0325ae4.png"

def main():
    st.title("Life Expectancy Prediction")
    st.markdown(
        f'<img src="{image_url}" style="width:800px; height:350px; object-fit:cover;">',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        st.header("Enter Specifications")
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox('Country Status', options=list(status_map.keys()))
            adult_mortality = st.number_input('Adult Mortality', min_value=1.0, max_value=500.0, value=200.0, step=1.0)
            infant_deaths = st.number_input('Infant Deaths', min_value=0, max_value=100, value=10, step=1)
            alcohol = st.number_input('Alcohol Consumption (liters per capita)', min_value=0.01, max_value=20.0, value=5.0, step=0.01)
            percentage_expenditure = st.number_input('Percentage Expenditure (GDP per capita)', min_value=0.0, max_value=1100.0, value=300.0, step=0.01)
            hepatitis_b = st.number_input('Hepatitis B immunization coverage (%)', min_value=50.0, max_value=100.0, value=80.0, step=1.0)
            measles = st.number_input('Measles cases per 1000 population', min_value=0, max_value=1000, value=235, step=1)
            bmi = st.number_input('BMI', min_value=0.0, max_value=80.0, value=30.0, step=0.1)
            under_five_deaths = st.number_input('Under-five Deaths', min_value=0, max_value=100, value=20, step=1)
        with col2:
            polio = st.number_input('Polio immunization coverage (%)', min_value=0.0, max_value=100.0, value=80.0, step=0.1)
            total_expenditure = st.number_input('Total expenditure (% of GDP)', min_value=0.0, max_value=15.0, value=6.0, step=1.0)
            diphtheria = st.number_input('Diphtheria immunization coverage (%)', min_value=0.0, max_value=100.0, value=80.0, step=0.1)
            hiv_aids = st.number_input('HIV/AIDS (deaths per 1000 live births)', min_value=0.0, max_value=5.0, value=0.5, step=0.01)
            gdp = st.number_input('GDP per capita', min_value=1.0, max_value=20000.0, value=5000.0, step=0.1)
            thinness_1_19_years = st.number_input('Thinness 1-19 years (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.01)
            thinness_5_9_years = st.number_input('Thinness 5-9 years (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.01)
            income_composition_of_resources = st.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.1, step = 0.01)
            schooling = st.number_input('Schooling (years)', min_value=0.0, max_value=20.0, value=10.0, step=1.0)
        lcol1, lcol2, lcol3 = st.columns([2, 1, 2])
        with lcol2:
            submitted = st.form_submit_button("Predict Price")
    if submitted:
        input_data = np.array([[status_map[status], adult_mortality, infant_deaths, alcohol, percentage_expenditure, hepatitis_b, measles, bmi, 
                        under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, thinness_1_19_years, thinness_5_9_years, 
                        income_composition_of_resources, schooling]])
        input_scaled = scaler.transform(input_data)
        predictedLifeExpectancy = model.predict(input_scaled)
        st.success(f'Predicted Life Expectancy: {predictedLifeExpectancy[0]:.2f} years')

if __name__ == "__main__":
    main()

