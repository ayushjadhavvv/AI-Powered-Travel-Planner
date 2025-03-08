import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_travel_query(query):
    template = (
        "You are an AI travel planner. Given the query below, provide travel recommendations "
        "including travel modes (cab, train, bus, flight) and estimated costs in a structured format.\n\n"
        "Query: {query}\n\n"
        "Answer:"
    )
    
    prompt = PromptTemplate(template=template, input_variables=["query"])

    llm = GoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.invoke({"query": query})

    return {"query": query, "text": response["text"]}  # Ensure it returns text correctly

