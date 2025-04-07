# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.recommender import get_top_k_recommendations

app = FastAPI(title="SHL Assessment Recommender")

# Input model for query
class QueryInput(BaseModel):
    query: str

# API endpoint to get recommendations
@app.post("/recommend")
def recommend_assessments(query_input: QueryInput):
    query = query_input.query
    results = get_top_k_recommendations(query)
    return {"recommendations": results}
