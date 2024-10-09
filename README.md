# AI-Enhanced Student Performance Predictor

## Overview

This Streamlit application provides an AI-enhanced tool for predicting and analyzing student performance. It uses machine learning models to predict student performance based on various factors and offers insights and recommendations for improvement.

## Features

- Data upload and preview
- Performance prediction using a pre-trained linear regression model
- AI-generated insights on student performance using Google's Generative AI
- Data visualizations including performance distribution and feature correlations
- Feature importance analysis
- Personalized recommendations for individual students
- PDF report generation
- Email functionality for sending reports to school authorities
- ngrok integration for tunneling (optional)

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- PyNGROK
- Google Generative AI
- ReportLab

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - NGROK_AUTH_TOKEN: Your ngrok authentication token (optional)
   - GOOGLE_API_KEY: Your Google Generative AI API key
   - EMAIL_ADDRESS: The email address for sending reports
   - EMAIL_PASSWORD: The password for the email account

   You can set these environment variables in your shell:

   ```
   export NGROK_AUTH_TOKEN="your_ngrok_auth_token"
   export GOOGLE_API_KEY="your_google_api_key"
   export EMAIL_ADDRESS="your_email@example.com"
   export EMAIL_PASSWORD="your_email_password"
   ```

   Or use a .env file with python-dotenv.

## Usage

1. Ensure all required environment variables are set.
2. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

3. Upload a CSV file containing student data.
4. Explore the predictions, insights, and visualizations provided by the app.
5. Generate and download PDF reports or send them via email to school authorities.

## Data Format

The input CSV file should contain the following columns:

- StudentID
- Age
- Gender
- ParentalEducation
- StudyTime
- Absences
- Tutoring
- ParentalSupport
- Sports
- Music
- Volunteering
- GradeLevel
- EconomicBackground
- Region
- LanguageBarrier
- TeacherStudentRatio
- BehavioralScore
- Overall_Percentage (optional)

## Model

The application uses a pre-trained linear regression model (`linear_regression_model.joblib`) for predicting student performance. Ensure this file is present in the same directory as the script.

## AI Insights

The application uses Google's Generative AI to provide insights and recommendations based on the predicted performance data. Ensure you have set up a valid GOOGLE_API_KEY in your environment variables.

## ngrok Integration

If you want to use ngrok for tunneling, ensure you have set the NGROK_AUTH_TOKEN environment variable. The application will use ngrok if the token is available.

## Customization

- Modify the `generate_recommendations` function to customize personalized recommendations.
- Adjust the PDF report generation in `generate_pdf_report` to change the report format or content.

## Security Note

Ensure that you keep your API keys and email credentials secure. Do not share them or commit them to version control. Always use environment variables or secure secret management systems to handle sensitive information.

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contact

For any queries or support, please contact Fikayo at [fikkyfresh81@gmail.com].
