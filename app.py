import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if the API key is loaded correctly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing! Please check your .env file.")

# Define function for processing travel query
def process_travel_query(query):
    template = (
        "You are an AI travel planner. Given the query below, provide travel recommendations "
        "including travel modes (cab, train, bus, flight) and estimated costs in a structured format.\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )
    
    prompt = PromptTemplate(template=template, input_variables=["query"])

    # Use a supported model
    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

    chain = LLMChain(llm=llm, prompt=prompt)

    # Use .invoke() instead of .run()
    response = chain.invoke({"query": query})
    return response

# Streamlit UI
st.title("AI-Powered Travel Planner")
st.write("Enter your source and destination to receive travel recommendations:")

source = st.text_input("Enter Source", "Mumbai")
destination = st.text_input("Enter Destination", "Delhi")
trip_date = st.date_input("Trip Date")

if st.button("Get Travel Recommendations"):
    user_query = f"Plan a trip from {source} to {destination} on {trip_date}. " \
                 "List available travel modes (cab, train, bus, flight) with estimated costs."
    
    st.write("Processing your query...")
    
    result = process_travel_query(user_query)
    
    st.markdown("### Travel Recommendations")
    st.markdown(result["text"])
