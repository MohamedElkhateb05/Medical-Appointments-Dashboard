🩺 Medical Appointments Dashboard


🎯 Project Overview
This is an interactive data dashboard built using Python, Dash, and Plotly to analyze the "Medical Appointment No Show" dataset. The primary goal of this project is to provide meaningful, data-driven insights into the factors that influence whether patients show up for their scheduled medical appointments.

📂 Dataset Information
Dataset Name: Medical Appointment No Show

The dataset contains over 110,000 medical appointment records from Brazil. It includes various features such as patient demographics (gender, age), appointment details (scheduled vs. appointment date), and health-related information (chronic conditions, SMS messages received).

✨ Features & Analysis
The dashboard is designed to be fully interactive, allowing users to explore the data dynamically. Key features and analyses include:

Key Metrics: Displays crucial metrics at a glance, such as the overall show-up rate, average wait time, and SMS reception rate.

Interactive Filters: Users can filter the data by age range, neighborhood, and specific medical conditions (e.g., diabetes, hypertension).

Visualizations: A set of charts provides deep insights:

Show-up rate by gender and day of the week.

A scatter plot showing the relationship between a patient's age and their wait time.

Distribution of appointments by neighborhood using a geospatial map.

Analysis of show-up rates based on different medical conditions.

⚙️ How to Run the Application
To run this dashboard on your local machine, follow these simple steps:

Clone the Repository:
Navigate to your desired directory and use the following command to clone this repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install Required Libraries:
Ensure you have Python installed. The project's dependencies are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

Run the Dashboard:
Finally, execute the main Python script to start the server:

python app.py

The dashboard will be accessible in your web browser at http://127.0.0.1:8050.
