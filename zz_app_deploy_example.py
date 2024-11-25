import numpy as np
from sklearn.linear_model import LinearRegression
import gradio  as gr

# Sample data for training the model
# X represents [house size (m²), number of rooms, house age]
X = np.array([
    [50, 1, 10],
    [80, 2, 5],
    [120, 3, 20],
    [150, 4, 15],
    [200, 4, 10]
])
# y represents house prices (in thousands of euros)
y = np.array([100, 150, 200, 250, 300])

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to make predictions based on user input
def predict_house_price(size, rooms, age):
    input_data = np.array([[size, rooms, age]])
    prediction = model.predict(input_data)
    return {"Predicted Price (in thousands of euros)": round(float(prediction[0]), 2)}

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
    fn=predict_house_price,
    inputs=[
        gr.Number(label="House Size (m²)"),
        gr.Number(label="Number of Rooms"),
        gr.Number(label="House Age (years)")
    ],
    outputs=gr.Textbox(label="Predicted Price (in thousands of euros)"),
    title="House Price Prediction",
    description="Enter the size, number of rooms, and age of the house to predict its price (in thousands of euros).",
    css=css,
    theme="default"
)

if __name__ == "__main__":
    iface.launch()
