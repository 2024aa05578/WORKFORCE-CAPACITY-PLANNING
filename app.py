
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.pyplot 

from optimizer import optimize_resources

# Load model
model = joblib.load('models/workforce_model.pkl')

st.set_page_config(page_title="AI Workforce Planning", layout="wide")

st.title("AI Enabled Workforce & Capacity Planning")

uploaded_file = st.file_uploader(
    "Upload Workforce CSV",
    type=['csv']
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month

    df['Location_Code'] = df['Location'].astype('category').cat.codes

    features = df[['Location_Code',
                   'Engineers',
                   'Skill_Level',
                   'Day',
                   'Month']]

    predictions = model.predict(features)

    df['Predicted_Workload'] = predictions

    st.subheader("Forecast Results")

    st.dataframe(df)

    # Graph
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df['Predicted_Workload'],
            marker='o',
            color='blue')

    ax.set_title("Predicted Demand")
    ax.set_xlabel("Records")
    ax.set_ylabel("Workload")

    st.pyplot(fig)

    # Optimization

    st.subheader("Resource Optimization")

    demand = {}

    for index, row in df.iterrows():
        demand[row['Location']] = round(row['Predicted_Workload'])

    total_engineers = st.number_input(
        "Available Engineers",
        value=50
    )

    if st.button("Optimize Allocation"):

        result = optimize_resources(
            demand,
            total_engineers
        )

        result_df = pd.DataFrame({
            'Location': result.keys(),
            'Allocated_Engineers': result.values()
        })

        st.dataframe(result_df)

        st.success("Optimization Completed")
