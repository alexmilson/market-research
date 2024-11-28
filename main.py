import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import InferenceClient

# Initialize DuckDuckGoSearchRun and Hugging Face client
search = DuckDuckGoSearchRun()
client = InferenceClient(api_key="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")

# Streamlit App
st.title("AI Use Case Generator with Hugging Face LLM")

# Input field for company name
company = st.text_input("Enter the Company Name:")

# Function to infer industry based on company name
def infer_industry(company):
    query = f"{company} industry"
    results = search.invoke(query)
    return results

# Function to generate AI/ML use cases based on the industry
def generate_use_cases_with_hf(industry):
    messages = [
        {
            "role": "user",
            "content": f"Suggest relevant AI and GenAI use cases for the {industry} industry, focusing on operations, supply chain, and customer experience. Highlight actionable insights and potential challenges."
        }
    ]
    
    response = ""
    with st.spinner("Generating use cases with Hugging Face LLM..."):
        stream = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            response += delta

    return response.strip()  # Clean up any leading/trailing whitespace

# Function to fetch relevant links (datasets or resources)
def fetch_relevant_links(company, industry):
    query = f"{company} {industry} datasets site:kaggle.com OR site:github.com"
    results = search.invoke(query)
    return results

# Streamlit Workflow
if st.button("Generate") and company:
    with st.spinner("Processing..."):
        try:
            # Step 1: Industry Insights
            st.subheader("Industry Insights")
            industry_info = infer_industry(company)
            st.write(industry_info)
            
            # Step 2: AI Use Cases
            st.subheader("AI/ML Use Cases")
            use_cases = generate_use_cases_with_hf(industry_info)
            st.write(use_cases)
            
            # Step 3: Dataset and Resource Links
            st.subheader("Relevant Datasets and Resources")
            links = fetch_relevant_links(company, industry_info)
            st.write(links)
            
            st.success("Data Generated Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
