import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Function to load the saved model and display results
def load_and_display_results(model, input_values):
    # Use the loaded model to make predictions
    input_data = pd.DataFrame([input_values])
    
    # Ensure column names match those used during training
    input_data = input_data[['industrial_risk', 'management_risk', 'financial_flexibility', 
                             'credibility', 'competitiveness', 'operating_risk']]

    y_pred = model.predict(input_data)

    return y_pred

# Streamlit UI
def main():
    st.title('Decision Tree Model Predictor')

    # Input values from the user
    st.subheader('Enter Values for Prediction:')
    industrial_risk = st.number_input('Industrial Risk', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    management_risk = st.number_input('Management Risk', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    financial_flexibility = st.number_input('Financial Flexibility', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    credibility = st.number_input('Credibility', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    competitiveness = st.number_input('Competitiveness', min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    operating_risk = st.number_input('Operating Risk', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    # Prepare input data as a dictionary
    input_values = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
    }

    # Load the trained model from pickle file
    model = pickle.load(open('FinalizedDc.pkl', 'rb'))
 # Predict and display results
    if st.button('Predict'):
        prediction = load_and_display_results(model, input_values)
        st.subheader('Prediction Result:')
        
        # Convert prediction to human-readable text
        if prediction[0] == 0:
            st.write("Predicted Class: Bankruptcy")
        else:
            st.write("Predicted Class: Non-Bankruptcy")

if __name__ == '__main__':
    main()

