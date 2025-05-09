from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
import uvicorn
from pydantic import BaseModel
from typing import List
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Movie Recommendation API",
    description="API for recommending similar movies based on user input",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow the new frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables
model_files = {
    "df": "model/movies_df.pkl",
    "similarity": "model/similarity_matrix.pkl"
}

# Response models
class MovieRecommendation(BaseModel):
    title: str
    movie_id: int

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    query: str
    matched_query: str

# Load model data function
def load_model_data():
    """Load the pre-trained model data from pickle files"""
    try:
        with open(model_files["df"], "rb") as f:
            df = pickle.load(f)
        with open(model_files["similarity"], "rb") as f:
            similarity = pickle.load(f)
        return df, similarity
    except FileNotFoundError:
        raise Exception("Model files not found. Please run the model training script first.")

# API routes
@app.get("/", response_model=dict)
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "Get movie recommendations",
            "/movies": "Get list of available movies"
        }
    }

@app.get("/recommend", response_model=RecommendationResponse)
def recommend_movies(movie: str = Query(..., description="Movie title to get recommendations for")):
    """Get movie recommendations based on input movie title"""
    df, similarity = load_model_data()
    
    # Preprocess the movie title: remove special characters and lowercase
    movie_processed = re.sub(r'[^\w\s]', '', movie).lower()
    
    # Find closest matching title in dataframe
    all_titles = df['title'].str.lower().tolist()
    closest_match = get_close_matches(movie_processed, all_titles, n=1, cutoff=0.6)
    
    if closest_match:
        movie_title = closest_match[0]  # Use closest match if found
    else:
        raise HTTPException(status_code=404, detail=f"Movie '{movie}' not found in the dataset.")
    
    # Get movie index
    movie_index = df[df['title'].str.lower() == movie_title].index[0]
    
    # Calculate similarities
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    # Format recommendations
    recommendations = []
    for i in movies_list:
        idx = i[0]
        recommendations.append(MovieRecommendation(
            title=df.iloc[idx].title,
            movie_id=int(df.iloc[idx].movie_id)
        ))
    
    return RecommendationResponse(
        recommendations=recommendations,
        query=movie,
        matched_query=df.iloc[movie_index].title
    )

@app.get("/movies", response_model=List[str])
def get_movies(limit: int = Query(100, description="Number of movies to return")):
    """Get list of movies available in the dataset"""
    df, _ = load_model_data()
    return df['title'].tolist()[:limit]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Check if model files exist on startup"""
    try:
        load_model_data()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: {e}")
        print("You need to train and save the model first.")

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)