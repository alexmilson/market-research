import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import InferenceClient

# Initialize DuckDuckGoSearchRun and Hugging Face client
search = DuckDuckGoSearchRun()
client = InferenceClient(api_key="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")

# Dictionary to store indexed research and use cases
research_index = {}
use_case_index = {}

# Streamlit App
st.title("AI Use Case Generator with Hugging Face LLM")

# Input field for company name (industry inferred)
company = st.text_input("Enter the Company Name:")

# Agent Workflow
def infer_industry(company):
    # Use a search engine to infer the industry based on the company name
    query = f"{company} industry"
    results = search.invoke(query)
    research_index["industry_info"] = results  # Storing industry info
    return results

def generate_use_cases_with_hf(industry):
    # Preparing input for the LLM
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
            st.write(delta, end="")

    use_case_index["use_cases"] = response  # Store the use case response in the dictionary
    return response

def fetch_relevant_links(company, industry):
    # Fetch links to relevant resources like Kaggle, GitHub, etc.
    query = f"{company} {industry} datasets site:kaggle.com OR site:github.com"
    results = search.invoke(query)
    return results

def format_search_results(raw_results):
    # Process raw search results for better formatting
    if not raw_results:
        return "No relevant information found for your query."
    
    formatted_output = f"""
    ## Search Results for Query:
    **Top Findings**: {raw_results}
    """
    return formatted_output

# Streamlit Workflow
if st.button("Generate"):
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
            
            st.success("Data Indexed Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Search functionality for indexed data
st.subheader("Search Indexed Data")
index_type = st.radio("Select Index to Search:", ("Research", "Use Cases"))
search_query = st.text_input("Enter your search query:")
if st.button("Search Index"):
    with st.spinner("Searching..."):
        try:
            # Select the correct index based on user selection
            index = research_index if index_type == "Research" else use_case_index
            search_results = format_search_results(index.get(search_query, ""))
            st.markdown(search_results, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
