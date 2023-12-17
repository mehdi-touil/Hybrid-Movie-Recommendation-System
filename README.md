# Movie Recommender System

## Overview

This project implements a versatile Movie Recommender System using various techniques such as Content-Based Filtering, Collaborative Filtering, and Deep Learning with TensorFlow. The system analyzes user preferences and provides personalized movie recommendations.

## Features

- **Content-Based Filtering:**
  - Utilizes movie metadata (genres, overview) to recommend movies based on user preferences.

- **Collaborative Filtering:**
  - Recommends movies by considering user behavior, leveraging historical ratings and preferences.

- **Hybrid Approach:**
  - Integrates thw Two approaches to create a best of both version

- **Deep Learning with TensorFlow:**
  - Integrates deep learning models for enhanced recommendation accuracy.
  - Uses TensorFlow Recommenders (TFRS) library for building and training recommendation models.

- **Streamlit Deployment:**
  - Interactive web-based interface for users to explore and receive movie recommendations.

## Project Structure

The project is structured as follows:

- **`data/`**: Contains dataset files.
- **`model.py`**: Holds model functions.
- **`*.pkl`**: Similarity files with data dumped.
- **`notebooks/`**: Jupyter notebooks for data exploration and model development.
- **`app.py`**: main Streamlit app file.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mehdi-touil/Movie-Recommender-System.git
   streamlit run app.py
