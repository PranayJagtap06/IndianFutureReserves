import datetime
import joblib
import pickle
import io
import csv
import numpy as np
import pandas as pd
import streamlit as st



model = joblib.load('foreign_reserves_prophet.joblib')
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# Setting up Streamlit app page
about = "Welcome to our Streamlit IndianFutureReserves application! This application is powered by a sophisticated machine learning model that predicts the future foreign exchange reserves of India in US $ million."\
    "Our model is trained on a comprehensive dataset sourced directly from the Reserve Bank of India, ensuring the accuracy and reliability of our predictions. The dataset includes a wide range of economic indicators, providing our model with a robust foundation for forecasting."\
    "The goal of this application is to provide users with an intuitive and interactive platform to explore and understand the dynamics of India's foreign exchange reserves. Whether you're an economist, a policy maker, a student, or just someone interested in the Indian economy, we believe this application will be a valuable tool for you."\
    "We're committed to making complex economic forecasting accessible and understandable. We hope you find this application insightful and useful in your endeavors. Enjoy exploring!"

st.set_page_config(page_title='IndianFutureReserves', page_icon='🤖', menu_items={'About': f"{about}"})
st.title(body="IndianFutureReserves: Predicting India's Financial Fortunes 🔮💰")
st.markdown("*Leverage the power of machine learning to unveil future trends in India's foreign exchange reserves with pinpoint precision, fueled by authoritative RBI data.*")

# columns = ['Date', 'Forward Premia of US$ 1-month (%)', 'Forward Premia of US$ 3-month (%)', 'Forward Premia of US$ 6-month (%)', 'Reverse Repo Rate (%)', 'Marginal Standing Facility (MSF) Rate (%)', 'Bank Rate (%)', 'Base Rate (%)', '91-Day Treasury Bill (Primary) Yield (%)', '182-Day Treasury Bill (Primary) Yield (%)', '364-Day Treasury Bill (Primary) Yield (%)', '10-Year G-Sec Yield (FBIL) (%)', 'Cash Reserve Ratio (%)', 'Statutory Liquidity Ratio (%)', 'Policy Repo Rate (%)', 'Standing Deposit Facility (SDF) Rate (%)']

st.markdown("""---""")
st.header("Select Parameters")
period = st.date_input('Date', min_value=datetime.date(2023, 11, 1), max_value=datetime.date(2074, 12, 31))
fp_1month = st.slider("Forward Premia of US$ 1-month (%)", min_value=1.00, max_value=100.00, value=7.25)
fp_3month = st.slider("Forward Premia of US$ 3-month (%)", min_value=1.00, max_value=100.00, value=4.15)
fp_6month = st.slider("Forward Premia of US$ 6-month (%)", min_value=1.00, max_value=100.00, value=3.25)
repo_rate = st.slider("Reverse Repo Rate (%)", min_value=1.00, max_value=100.00, value=5.50)
msf_rate = st.slider("Marginal Standing Facility (MSF) Rate (%)", min_value=1.00, max_value=100.00, value=6.25)
bank_rate = st.slider("Bank Rate (%)", min_value=1.00, max_value=100.00, value=5.15)
base_rate = st.slider("Base Rate (%)", min_value=1.00, max_value=100.00, value=8.00)
tby_91 = st.slider("91-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=6.15)
tby_182 = st.slider("182-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=4.63)
tby_364 = st.slider("364-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=6.35)
fbil = st.slider("10-Year G-Sec Yield (FBIL) (%)", min_value=1.00, max_value=100.00, value=5.48)
crr = st.slider("Cash Reserve Ratio (%)", min_value=1.00, max_value=100.00, value=3.65)
slr = st.slider("Statutory Liquidity Ratio (%)", min_value=1.00, max_value=100.00, value=19.56)
prr = st.slider("Policy Repo Rate (%)", min_value=1.00, max_value=100.00, value=5.45)
sdf = st.slider("Standing Deposit Facility (SDF) Rate (%)", min_value=1.00, max_value=100.00, value=4.56)
predict = st.button('Predict')
history = st.button('History')
delete_history = st.button('Delete History')
download_history = st.button("Download History")

# Initialize session state for session history
if "session_history" not in st.session_state:
    st.session_state.session_history = []

# Creating DataFrame
df = pd.DataFrame(data={'ds': [period], 'Forward Premia of US$ 1-month (%)': [fp_1month],
                        'Forward Premia of US$ 3-month (%)': [fp_3month], 'Forward Premia of US$ 6-month (%)': [fp_6month],
                        'Reverse Repo Rate (%)': [repo_rate], 'Marginal Standing Facility (MSF) Rate (%)': [msf_rate],
                        'Bank Rate (%)': [bank_rate], 'Base Rate (%)': [base_rate],
                        '91-Day Treasury Bill (Primary) Yield (%)': [tby_91], '182-Day Treasury Bill (Primary) Yield (%)': [tby_182],
                        '364-Day Treasury Bill (Primary) Yield (%)': [tby_364], '10-Year G-Sec Yield (FBIL) (%)': [fbil],
                        'Cash Reserve Ratio (%)': [crr], 'Statutory Liquidity Ratio (%)': [slr],
                        'Policy Repo Rate (%)': [prr], 'Standing Deposit Facility (SDF) Rate (%)': [sdf]})

# Creating a function to predict foreign reserves
def predictreserves(dataframe: object, model_: object):
    # Predict the future foreign reserves
    pred_ = model_.predict(dataframe)

    return pred_['yhat'], dataframe['ds'][0], dataframe

# Writting History function
def display_history():
    st.markdown("""---""")
    st.header("Session History")
    for i, item in enumerate(st.session_state.session_history[::-1]):
        df_ = item['dataframe']
        _pred = item['prediction']

        st.write('Dataframe: ')
        st.write(df_)
        st.write('---')
        st.write('Prediction: ')
        st.write(_pred)
        st.write('---')
        st.write('---')

# If Predict button is pressed generate result and print
if predict:
    result_, date_, df_ = predictreserves(dataframe=df, model_=model)

    # Write prediction in Streamlit
    st.markdown("""---""")
    st.header("Prediction")
    txt = f"Foreign Reserves on {date_} will be ${np.round(result_[0], 2):,} Million."
    st.subheader(txt)

    # Renaming 'ds' column in dataframe to 'Date'
    df_.rename(columns={'ds': 'Period'}, inplace=True)

    # Add dataframe and prediction to session history
    st.session_state.session_history.append({'dataframe': df_, 'prediction': txt})

if history:
    display_history()

if delete_history:
    st.session_state.session_history.clear()
    display_history()

if download_history:
    # Create a CSV file from the session history
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Period", "Forward Premia of US$ 1-month (%)", "Forward Premia of US$ 3-month (%)", "Forward Premia of US$ 6-month (%)",
                    "Reverse Repo Rate (%)", "Marginal Standing Facility (MSF) Rate (%)", "Bank Rate (%)", "Base Rate (%)",
                    "91-Day Treasury Bill (Primary) Yield (%)", "182-Day Treasury Bill (Primary) Yield (%)", "364-Day Treasury Bill (Primary) Yield (%)",
                    "10-Year G-Sec Yield (FBIL) (%)", "Cash Reserve Ratio (%)", "Statutory Liquidity Ratio (%)",
                    "Policy Repo Rate (%)", "Standing Deposit Facility (SDF) Rate (%)", "Prediction"])
    for item in st.session_state.session_history:
        writer.writerow([item["dataframe"]["Period"][0]] + list(item["dataframe"].iloc[0, 1:]) + [item["prediction"]])

    # Download the CSV file
    st.download_button(
        label="click to download",
        data=csv_buffer.getvalue(),
        file_name="session_history.csv",
        mime="text/csv",
    )

st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap06.github.io)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_sq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# st.image("assets/pranay_sq.jpg", width=125)
st.markdown("Electrical Engineer ⚡ | Python enthusiast 🐍 | Machine learning explorer 🤖 "\
            "<br>📍 Nagpur, Maharashtra, India", unsafe_allow_html=True)
