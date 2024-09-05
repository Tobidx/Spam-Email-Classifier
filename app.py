import gradio as gr
import joblib
import xgboost as xgb
import numpy as np

def classify_email(email_text):
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_model.joblib')
    email_tfidf = tfidf.transform([email_text])
    email_dmatrix = xgb.DMatrix(email_tfidf)
    prediction = model.predict(email_dmatrix)[0]
    confidence = max(prediction, 1 - prediction)
    label = "Spam" if prediction > 0.5 else "Not Spam"
    return {label: float(confidence)}

def analyze_email(email_text):
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('spam_model.joblib')
    email_tfidf = tfidf.transform([email_text])
    email_dmatrix = xgb.DMatrix(email_tfidf)
    prediction = model.predict(email_dmatrix)[0]
    confidence = max(prediction, 1 - prediction)
    label = "Spam" if prediction > 0.5 else "Not Spam"
    
    # Get feature importance
    feature_names = tfidf.get_feature_names_out()
    feature_importance = model.get_score(importance_type='gain')
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    analysis = f"Classification: {label} (Confidence: {confidence:.2%})\n\n"
    analysis += "Top 5 influential words:\n"
    for feature, importance in top_features:
        if feature in email_text.lower():
            analysis += f"- {feature}: {importance:.2f}\n"
    
    return analysis

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as iface:
    gr.Markdown(
    """
    # 🚀  Spam Email Classifier
    
    Using Machine Learning to detect spam emails with high accuracy!
    """
    )
    with gr.Row():
        with gr.Column(scale=2):
            email_input = gr.Textbox(lines=5, label="Enter email text")
            with gr.Row():
                classify_btn = gr.Button("Classify")
                analyze_btn = gr.Button("Detailed Analysis")
        with gr.Column(scale=1):
            label_output = gr.Label(label="Classification")
            analysis_output = gr.Textbox(label="Detailed Analysis", lines=8)
    
    examples = [
        ["Get fat quick! Buy our cheese burger now!"],
        ["Hi Ajibola, let's go out on a date tonight"],
        ["Congratulations! You've won a free iPhone. Click here to claim."],
        ["Please find attached the report for Q2 sales figures."]
    ]
    gr.Examples(examples, inputs=email_input)
    
    classify_btn.click(classify_email, inputs=email_input, outputs=label_output)
    analyze_btn.click(analyze_email, inputs=email_input, outputs=analysis_output)
    
    gr.Markdown(
    """
    ### How it works
    This classifier uses an XGBoost model trained on a large dataset of over 190,000 emails.
    The model achieved a 98% accuracy on the training data and 94% accuracy on the test data.
    It analyzes the content and structure of the email to determine if it's spam or not.
    
    ### Tips for use
    - Enter the full text of the email for best results
    - The 'Detailed Analysis' shows the top words influencing the classification
    - Confidence score indicates how sure the model is about its prediction
    """
    )

# Launch the interface
iface.launch()
