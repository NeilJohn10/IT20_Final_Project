import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HR Talent Predictor", page_icon="🚀", layout="centered")

# --- 2. LOAD ASSETS (Cached so it runs fast) ---
@st.cache_resource
def load_components():
    model = joblib.load('final_random_forest_promotion_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

model, scaler, selected_features = load_components()

# --- 3. DICTIONARIES FOR CATEGORICAL MAPPING ---
dept_mapping = {
    "Engineering": 0, "Finance": 1, "HR": 2, "Marketing": 3, 
    "Operations": 4, "Sales": 5, "Support": 6
}
gender_mapping = {"Female": 0, "Male": 1}
edu_mapping = {"Bachelor": 0, "Master": 1, "PhD": 2}


# --- NEW: 4. MODAL DIALOG FOR HISTORY ---
# This creates a large, center-screen popup specifically for wide tables
@st.dialog("🗄️ System Prediction History", width="large")
def show_history_modal():
    try:
        conn = sqlite3.connect('local_history.db')
        df_history = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY Timestamp DESC", conn)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        conn.close()
        
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='hr_prediction_history.csv',
            mime='text/csv',
            use_container_width=True
        )
    except Exception:
        st.info("No history found yet. Make a prediction to start logging!")


# --- 5. SYSTEM HEADER & HISTORY BUTTON ---
header_col1, header_col2 = st.columns([0.75, 0.25])

with header_col1:
    st.title("🚀 HR Predictor")

with header_col2:
    st.write("") 
    st.write("")
    # This acts as a normal button that triggers the modal above
    if st.button("🗄️ View History", use_container_width=True):
        show_history_modal()

st.markdown("""
Welcome to the AI-driven shortlisting tool. 
Enter the employee's metrics below to evaluate their readiness for promotion based on historical company data.
""")
st.divider()

# --- 6. DATA INPUT FORM ---
with st.form("employee_data_form"):
    st.subheader("Employee Performance Metrics")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    for i, feature in enumerate(selected_features):
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            # Categorical Dropdowns
            if feature == 'department':
                selected_dept = st.selectbox("Department", options=list(dept_mapping.keys()))
                input_data[feature] = dept_mapping[selected_dept]
                
            elif feature == 'gender':
                selected_gen = st.selectbox("Gender", options=list(gender_mapping.keys()))
                input_data[feature] = gender_mapping[selected_gen]
                
            elif feature == 'education_level':
                selected_edu = st.selectbox("Education Level", options=list(edu_mapping.keys()))
                input_data[feature] = edu_mapping[selected_edu]

            # Numerical Sliders (Scores 1 to 5)
            elif feature in ['performance_score', 'performance_last_year', 'performance_two_years_ago', 'manager_rating', 'peer_feedback_score']:
                display_name = feature.replace('_', ' ').title()
                input_data[feature] = st.slider(display_name, min_value=1.0, max_value=5.0, value=3.0, step=0.1)

            # Numerical Sliders (Percentages 0 to 100)
            elif feature in ['kpi_achievement_percent', 'employee_engagement_score', 'job_satisfaction_score']:
                display_name = feature.replace('_', ' ').title()
                input_data[feature] = st.slider(display_name, min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                
            # Fallback for anything else (Tasks completed, age, etc.)
            else:
                display_name = feature.replace('_', ' ').title()
                if feature != 'employee_id': 
                    input_data[feature] = st.number_input(display_name, min_value=0.0, value=10.0)

    st.divider()
    submit_button = st.form_submit_button(label="🔍 Evaluate Candidate")

# --- 7. PREDICTION LOGIC & DB SAVE ---
if submit_button:
    input_df = pd.DataFrame([input_data])
    input_df = input_df[selected_features]
    
    try:
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)
        
        result_text = "PROMOTE" if prediction[0] == 1 else "DO NOT PROMOTE YET"
        
        if prediction[0] == 1:
            st.success(f"### ✅ Recommendation: {result_text}")
            st.balloons()
            st.markdown("This candidate's metrics strongly align with historical top-performers. They are highly recommended for the next promotion cycle.")
        else:
            st.warning(f"### ⏸️ Recommendation: {result_text}")
            st.markdown("This candidate does not currently meet the threshold for promotion. Consider assigning them to further training or leadership development programs.")
            
        # LOCAL DATABASE SAVE
        history_df = input_df.copy()
        history_df['Prediction_Result'] = result_text
        history_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect('local_history.db')
        history_df.to_sql('prediction_history', conn, if_exists='append', index=False)
        conn.close()
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")