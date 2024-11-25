import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

def generate_weekday_dataframe(week_day):
    days = ["Week_Day_Mon", "Week_Day_Tue", "Week_Day_Wed", "Week_Day_Thu", 
            "Week_Day_Fri", "Week_Day_Sat", "Week_Day_Sun"]

    day_mapping = {
        "monday": "Week_Day_Mon",
        "tuesday": "Week_Day_Tue",
        "wednesday": "Week_Day_Wed",
        "thursday": "Week_Day_Thu",
        "friday": "Week_Day_Fri",
        "saturday": "Week_Day_Sat",
        "sunday": "Week_Day_Sun",
    }

    week_day_lower = week_day.lower()
    if week_day_lower not in day_mapping:
        raise ValueError(f"Invalid week day: {week_day}. Please provide a valid day.")

    df = pd.DataFrame([{
        day: 1 if day == day_mapping[week_day_lower] else 0
        for day in days
    }])
    
    return df[['Week_Day_Fri', 'Week_Day_Sat', 'Week_Day_Thu', 'Week_Day_Wed']]

def generate_month_dataframe(month_in):
    months = ['month_jan', 'month_feb', 'month_mar', 'month_apr', 'month_may', 'month_jun',
            'month_jul', 'month_aug', 'month_sep', 'month_oct', 'month_nov', 'month_dec']

    month_mapping = {
    "january": "month_jan",
    "february": "month_feb",
    "march": "month_mar",
    "april": "month_apr",
    "may": "month_may",
    "june": "month_jun",
    "july": "month_jul",
    "august": "month_aug",
    "september": "month_sep",
    "october": "month_oct",
    "november": "month_nov",
    "december": "month_dec"
    }

    month_lower = month_in.lower()
    if month_lower not in month_mapping:
        raise ValueError(f"Invalid month name: {month_in}. Please provide a valid month.")

    df = pd.DataFrame([{
        month: 1 if month == month_mapping[month_lower] else 0
        for month in months
    }])
    
    return df[['month_apr', 'month_aug', 'month_jul', 'month_mar', 'month_may', 'month_nov', 'month_oct']]

def process_input(column, boolean_value):
    value = 1 if boolean_value else 0
    return pd.DataFrame([value], columns=[column])

def process_education(input_choices):
    result = {
        "education_secondary": 0,
        "education_tertiary": 0
        }
    if "Secondary".lower() in input_choices:
        result["education_secondary"] = 1
    if "Tertiary".lower() in input_choices:
        result["education_tertiary"] = 1
    
    return pd.DataFrame([result])

# Function to make predictions based on user input
def predict_success(campaign, cellular, duration, education,
                        housing, loan, job_collar, marital, previous, 
                        poutcome, age, day, month, day_of_week):
    data = []
    
    campaign = process_input('campaign',campaign)
    cellular = process_input('contact_cellular', cellular)
    education = process_education(education)
    housing = process_input('housing', housing)
    loan = process_input('loan', loan)
    job_collar = process_input('job_blue-collar', job_collar)
    marital = process_input('marital_single', marital)
    poutcome = process_input('poutcome_success', poutcome)
    month = generate_month_dataframe(month)
    day_of_week = generate_weekday_dataframe(day_of_week)
    
    data = pd.concat([campaign, cellular, education, housing, loan,
                      job_collar, marital, poutcome, month, day_of_week], axis=1)

    data['duration'] = duration
    data['previous'] = previous
    data['age'] = age
    data['day'] = day

    data = data[['Week_Day_Fri', 'Week_Day_Sat', 'Week_Day_Thu', 'campaign',
       'contact_cellular', 'day', 'duration', 'education_secondary',
       'education_tertiary', 'housing', 'job_blue-collar', 'loan',
       'marital_single', 'month_apr', 'month_aug', 'month_jul', 'month_mar',
       'month_may', 'month_nov', 'month_oct', 'poutcome_success', 'previous',
       'age', 'Week_Day_Wed']].copy()
     
    train = pd.read_parquet(r'Datasets\\parquet\\bank_feature_selection.parquet')
    X2_train = train.drop(['y'], axis = 1)
    y2_train = train['y']

    scaler = MinMaxScaler().fit(X2_train)
    test = pd.DataFrame(scaler.transform(data), columns = data.columns, index = data.index)

    final_model_gb = GradientBoostingClassifier(learning_rate = 0.01,
                                            max_depth = 5,
                                            max_features = 19, 
                                            min_samples_leaf = 50,
                                            min_samples_split = 2,
                                            n_estimators = 500,
                                            random_state = 99)
    

    final_model = final_model_gb.fit(X2_train, y2_train)
    final_model.predict(test)
    predict_proba_test = final_model.predict_proba(test)
    predict_proba_test

    final_pred = []

    for value in predict_proba_test[:,1]:
       if (value>=0.491400):
           final_pred.append(1)
       else:
         final_pred.append(0)

    prediction = "Success" if final_pred == 1 else "Failure"
    return prediction

# Custom CSS to adjust font sizes and text colors
css = """
body {
    background-color: white !important;
    color: black !important;
}

h1 {
    font-size: 32px !important;
    font-weight: bold !important;
    text-align: center !important;
    color: black !important;
}

label {
    font-size: 18px !important;
    color: black !important;
}

input[type="number"] {
    font-size: 18px !important;
    color: black !important;
    background-color: white !important;
}

textarea {
    font-size: 18px !important;
    color: black !important;
    background-color: white !important;
}

.gradio-container {
    font-size: 16px !important;
    color: black !important;
    background-color: white !important;
}

p {
    font-size: 16px !important;
    color: black !important;
}
"""

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_success,
    inputs=[
        gr.Number(label="Number of contacts performed during this campaign for this client:"),
        gr.Checkbox(label="Contact made by cellular?"),
        gr.Number(label="Last contact duration, in seconds"),
        gr.Checkbox(
            label="Day of the Week of last contact",
            choices=["Secondary", "Tertiary", "other"]
        ),
        gr.Checkbox(label="Has a housing loan?"),
        gr.Checkbox(label="Has a personal loan?"),
        gr.Checkbox(label="Blue collar job?"),
        gr.Checkbox(label="Is the client single?"),
        gr.Number(label="Number of contacts performed before this campaign and for this client"),
        gr.Checkbox(label="If the client was targeted in a previous campaign was it a success?"),
        gr.Number(label="Last contact day of the month",
                minimum=1, 
                maximum=31,
                step=1),
        gr.Dropdown(
            label="Month",
            choices=['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
        ),
        gr.Dropdown(
            label="Day of the Week of last contact",
            choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    ],
    outputs=gr.Textbox(label="Predicted outcome of the campaign for the client"),
    title="Bank marketing outcome prediction",
    #description="Enter the size, number of rooms, age of the house, and select the day of the week to predict its price (in thousands of euros).",
    css=css,
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
