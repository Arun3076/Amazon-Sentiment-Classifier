#python -m streamlit run app.py
import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Amazon Review Sentiment Classifier",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model from local fine-tuned folder
@st.cache_resource
def load_pipeline():
    try:
        model_dir = "amazon-model"
        return pipeline("text-classification", model=model_dir, tokenizer=model_dir)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

clf = load_pipeline()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for history
with st.sidebar:
    st.header("🕒 Analysis History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[["Timestamp", "Review", "Sentiment", "Confidence"]].style.format({"Confidence": "{:.2f}"}))
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main UI
st.title("🛒 Amazon Review Sentiment Classifier")
st.markdown("""
Analyze the sentiment of Amazon product reviews. Enter a review below to classify it as **Positive**, **Negative**, or **Neutral**.
""")

# Input area
review = st.text_area("✍️ Your Review:", height=150, placeholder="e.g., This product exceeded my expectations! Fast delivery and great quality.")

# Predict button
if st.button("🧠 Predict Sentiment", type="primary"):
    if not clf:
        st.error("Model not loaded. Please check the model directory.")
    elif review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = clf(review)[0]
                label = result['label']
                confidence = result['score']
                
                # Map labels to human-readable sentiment
                sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
                sentiment = sentiment_map.get(label, "Unknown")
                
                # Color-coded sentiment display
                if sentiment == "Positive":
                    st.success(f"**Sentiment:** {sentiment} 😊 \n\n **Confidence:** {confidence:.2f}")
                elif sentiment == "Negative":
                    st.error(f"**Sentiment:** {sentiment} 😔 \n\n **Confidence:** {confidence:.2f}")
                elif sentiment == "Neutral":
                    st.info(f"**Sentiment:** {sentiment} 😐 \n\n **Confidence:** {confidence:.2f}")
                else:
                    st.warning(f"**Unknown Sentiment:** {label} \n\n **Confidence:** {confidence:.2f}")
                
                # Add to history
                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Review": review[:50] + "..." if len(review) > 50 else review,
                    "Sentiment": sentiment,
                    "Confidence": confidence
                })
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")



# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Transformers | Model: amazon-model")


# # Display enhanced quick stats
# if st.session_state.history:
#     st.markdown("---")
#     st.subheader("📊 Sentiment Analysis Dashboard")
#     history_df = pd.DataFrame(st.session_state.history)
    
#     # Metrics layout
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Total Reviews", len(history_df))
#     col2.metric("Positive 😊", len(history_df[history_df["Sentiment"] == "Positive"]), f"{(len(history_df[history_df['Sentiment'] == 'Positive']) / len(history_df) * 100):.1f}%")
#     col3.metric("Negative 😔", len(history_df[history_df["Sentiment"] == "Negative"]), f"{(len(history_df[history_df['Sentiment'] == 'Negative']) / len(history_df) * 100):.1f}%")
#     col4.metric("Neutral 😐", len(history_df[history_df["Sentiment"] == "Neutral"]), f"{(len(history_df[history_df['Sentiment'] == 'Neutral']) / len(history_df) * 100):.1f}%")
    
#     # Pie chart for sentiment distribution
#     sentiment_counts = history_df["Sentiment"].value_counts().reset_index()
#     sentiment_counts.columns = ["Sentiment", "Count"]
#     fig = px.pie(
#         sentiment_counts,
#         values="Count",
#         names="Sentiment",
#         title="Sentiment Distribution",
#         color="Sentiment",
#         color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B", "Neutral": "#636EFA"}
#     )
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Download history as CSV
#     csv = history_df.to_csv(index=False)
#     st.download_button(
#         label="📥 Download History as CSV",
#         data=csv,
#         file_name="amazon_sentiment_history.csv",
#         mime="text/csv"
#     )
# Display quick stats
# if st.session_state.history:
#     st.markdown("---")
#     st.subheader("📊 Quick Stats")
#     history_df = pd.DataFrame(st.session_state.history)
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Reviews", len(history_df))
#     col2.metric("Positive", len(history_df[history_df["Sentiment"] == "Positive"]))
#     col3.metric("Negative", len(history_df[history_df["Sentiment"] == "Negative"]))
