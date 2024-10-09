import os
import smtplib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
from pyngrok import ngrok
import os
import joblib
from google.colab import userdata
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import textwrap
from IPython.display import Markdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# Retrieve API keys and tokens from environment variables
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Error: GOOGLE_API_KEY not set. AI-generated insights will not be available.")

def to_markdown(text):
    text = text.replace('•', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Initialize Gemini model
model_name = 'models/gemini-1.5-flash-8b'
available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
if model_name not in available_models:
    raise ValueError(f"Model {model_name} is not available in the list of models.")
gen_model = genai.GenerativeModel(model_name)

def get_ai_insights(data, predictions):
    data_summary = data.describe().to_string()
    prompt = f"""
    As an AI education expert, analyze this student performance data:

    Data Summary:
    {data_summary}

    Predictions Summary:
    Mean predicted performance: {predictions.mean():.2f}
    Max predicted performance: {predictions.max():.2f}
    Min predicted performance: {predictions.min():.2f}

    Please provide:
    1. Key insights about the predictions
    2. Potential factors influencing student performance
    3. Recommendations for improvement
    Keep the response concise and actionable.
    """

    response = gen_model.generate_content(prompt)
    return response.text

def generate_educational_email_content(data, predictions, ai_insights):
    performance_brackets = {
        'Exceptional (90-100%)': len(predictions[predictions >= 90]),
        'High (80-89%)': len(predictions[(predictions >= 80) & (predictions < 90)]),
        'Average (70-79%)': len(predictions[(predictions >= 70) & (predictions < 80)]),
        'Below Average (60-69%)': len(predictions[(predictions >= 60) & (predictions < 70)]),
        'Needs Improvement (<60%)': len(predictions[predictions < 60])
    }

    correlations = {}
    for column in data.columns:
        if column not in ['StudentID', 'Predicted_Performance']:
            correlation = data[column].corr(data['Predicted_Performance'])
            correlations[column] = abs(correlation)

    top_factors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:3]

    prompt = f"""
    As an educational data scientist writing to school authorities, compose a comprehensive email about student performance predictions and improvement strategies.

    Data Overview:
    - Total number of students analyzed: {len(predictions)}
    - Average predicted performance: {predictions.mean():.2f}%
    - Performance distribution:
      {', '.join([f"{k}: {v} students" for k, v in performance_brackets.items()])}

    Key Correlating Factors:
    {', '.join([f"{factor}: {corr:.2f}" for factor, corr in top_factors])}

    Original AI Insights:
    {ai_insights}

    Please write a formal email that includes:

    1. A professional introduction explaining the purpose and importance of this analysis
    2. Key findings section highlighting:
       - Overall performance trends
       - Specific areas of concern
       - Positive outcomes and areas of success
    3. Data-driven insights section:
       - Analysis of the factors most strongly correlated with student performance
       - Patterns or trends identified in the data
    4. Detailed recommendations:
       - Targeted interventions for different student performance brackets
       - Specific, actionable strategies for improvement
       - Resource allocation suggestions based on the findings
    5. Implementation roadmap:
       - Short-term actions (next 30 days)
       - Medium-term strategies (1-3 months)
       - Long-term goals (3-6 months)
    6. Offer for a follow-up meeting to discuss the findings and recommendations in detail

    The tone should be professional, data-driven, and solution-oriented. Emphasize the collaborative nature of improving student outcomes.
    """

    response = gen_model.generate_content(prompt)
    return response.text

def encode_string_columns(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(receiver_email, subject, body, attachment=None):
    global EMAIL_ADDRESS, EMAIL_PASSWORD
    
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        st.error("Email credentials are not set. Please set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = receiver_email
    msg['Subject'] = subject

    html_body = f"""
    <html>
      <head></head>
      <body>
        <pre style="font-family: Arial, sans-serif; white-space: pre-wrap;">
{body}
        </pre>
      </body>
    </html>
    """

    msg.attach(MIMEText(body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    if attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename=detailed_student_performance_analysis.csv')
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        st.success("Report sent successfully to school authorities!")
    except Exception as e:
        st.error(f"Failed to send report: {str(e)}")

# 1. Data Visualization Enhancements
def plot_performance_distribution(data):
    fig = px.histogram(data, x='Predicted_Performance', nbins=20,
                       title='Distribution of Predicted Student Performance')
    fig.update_layout(xaxis_title='Predicted Performance', yaxis_title='Count')
    st.plotly_chart(fig)

def plot_feature_correlations(data):
    corr = data.corr()
    fig = px.imshow(corr, title='Feature Correlations')
    st.plotly_chart(fig)

# 3. Feature Importance Analysis
def analyze_feature_importance(model, X, y):
    # Exclude 'Predicted_Performance' from features
    features_for_importance = X.drop(columns=['Predicted_Performance'], errors='ignore')
    
    perm_importance = permutation_importance(model, features_for_importance, y)
    feature_importance = pd.DataFrame({
        'feature': features_for_importance.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(feature_importance, x='feature', y='importance',
                 title='Feature Importance Analysis')
    st.plotly_chart(fig)
    return feature_importance

# 6. Personalized Recommendations
def generate_recommendations(student_data, feature_importance):
    recommendations = []
    top_features = feature_importance.head(3)['feature'].tolist()
    
    for feature in top_features:
        if student_data[feature] < student_data[feature].mean():
            recommendations.append(f"Focus on improving {feature}")
    
    return recommendations

# 8. PDF Report Generation
def generate_pdf_report(data, predictions, ai_insights, feature_importance):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Student Performance Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Summary Statistics
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    summary_data = [
        ["Metric", "Value"],
        ["Average Predicted Performance", f"{predictions.mean():.2f}%"],
        ["Highest Predicted Performance", f"{predictions.max():.2f}%"],
        ["Lowest Predicted Performance", f"{predictions.min():.2f}%"]
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # AI Insights
    story.append(Paragraph("AI-Generated Insights", styles['Heading2']))
    story.append(Paragraph(ai_insights, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Feature Importance
    story.append(Paragraph("Top Influencing Factors", styles['Heading2']))
    top_features = feature_importance.head(5).values.tolist()
    feature_table = Table([["Feature", "Importance"]] + top_features)
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(feature_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("AI-Enhanced Student Performance Predictor")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    st.write("Data Preview:", data.head())

    data, label_encoders = encode_string_columns(data)

    model_path = 'linear_regression_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.write("Model loaded successfully!")

        if 'Overall_Percentage' in data.columns:
            features = data.drop(columns=['Overall_Percentage'], errors='ignore')
        else:
            features = data

        expected_features = model.feature_names_in_
        missing_features = set(expected_features) - set(features.columns)
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
        else:
            predictions = model.predict(features)
            data['Predicted_Performance'] = predictions

            # Get AI insights
            with st.spinner("Generating AI insights..."):
                ai_insights = get_ai_insights(data, predictions)

            # Display predictions and insights
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predicted Performance")
                st.write(data[['StudentID', 'Predicted_Performance']])

            with col2:
                st.subheader("AI Insights")
                st.markdown(ai_insights)

            # Visualize the predictions
            fig, ax = plt.subplots()
            ax.bar(data['StudentID'], data['Predicted_Performance'], color='skyblue')
            ax.set_xlabel('Student ID')
            ax.set_ylabel('Predicted Performance (%)')
            ax.set_title('Predicted Performance of Students')
            st.pyplot(fig)

                # 1. Data Visualization Enhancements
            st.subheader("Data Visualizations")
            plot_performance_distribution(data)
            plot_feature_correlations(data)

            # 3. Feature Importance Analysis
            st.subheader("Feature Importance Analysis")
            if 'Predicted_Performance' in data.columns:
                feature_importance = analyze_feature_importance(model, features, data['Predicted_Performance'])
            else:
                st.error("'Predicted_Performance' column not found in the data.")

            # 6. Personalized Recommendations
            st.subheader("Personalized Recommendations")
            student_id = st.selectbox("Select a student ID for personalized recommendations:", data['StudentID'].unique())
            student_data = data[data['StudentID'] == student_id].iloc[0]
            recommendations = generate_recommendations(student_data, feature_importance)
            for rec in recommendations:
                st.write(f"- {rec}")

            # 8. PDF Report Generation
            st.subheader("Generate PDF Report")
            if st.button("Generate PDF Report"):
                pdf_buffer = generate_pdf_report(data, predictions, ai_insights, feature_importance)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="student_performance_report.pdf",
                    mime="application/pdf"
                )

            # Email functionality
            st.subheader("Generate Report for School Authorities")
            receiver_email = st.text_input("Enter school authority's email:")

            if st.button("Generate Comprehensive Report"):
                with st.spinner("Generating detailed educational report..."):
                    email_body = generate_educational_email_content(data, predictions, ai_insights)
                    st.subheader("Generated Report Preview")
                    st.text_area("Report Content", email_body, height=400)

            if st.button("Send Report to School Authorities"):
                if receiver_email:
                    detailed_csv = io.StringIO()
                    detailed_data = data.copy()
                    detailed_data['Performance_Category'] = pd.cut(
                        detailed_data['Predicted_Performance'],
                        bins=[0, 60, 70, 80, 90, 100],
                        labels=['Needs Improvement', 'Below Average', 'Average', 'High', 'Exceptional']
                    )
                    detailed_data.to_csv(detailed_csv, index=False)
                    detailed_csv.seek(0)

                    if 'email_body' not in locals():
                        email_body = generate_educational_email_content(data, predictions, ai_insights)

                    send_email(
                        receiver_email,
                        "Comprehensive Student Performance Analysis and Improvement Strategies",
                        email_body,
                        attachment=detailed_csv
                    )
                else:
                    st.error("Please enter the school authority's email address.")
    else:
        st.error("Model file not found. Please make sure the model file is in the path specified.")