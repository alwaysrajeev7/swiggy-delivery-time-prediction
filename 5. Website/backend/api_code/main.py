"""
DeliveryETA — FastAPI Backend
=============================

This module serves as the entry point for the DeliveryETA API.

Responsibilities:
    1. Initialize the FastAPI application.
    2. Expose HTTP endpoints for interacting with the application.
    3. Receive and validate delivery information using Pydantic.
    4. Pass validated information to the trained ML model.
    5. Return the predicted delivery time as a JSON response.
    6. Handle unexpected errors during model inference.

Endpoints:
    GET  /           -> Welcome message
    GET  /health     -> Health check
    POST /predict    -> Predict food delivery time

Note:
    The actual Machine Learning prediction logic is implemented
    separately in ml_model/model_prediction.py """


# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================
import logging
from datetime import datetime as dt
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Pydantic schema for validating incoming delivery information
from backend.schema.data_validation import Order_Details

# Function responsible for generating predictions 
# using the saved Machine Learning model.
from backend.ml_model.model_prediction import model, MODEL_VERSION, predict_delivery_time

# ==========================================================
# 2. INITIALIZE FASTAPI APPLICATION
# ==========================================================

des = 'REST API for predicting food delivery time using Gradient Boosting Regression model.'

# Fast API Object
app = FastAPI(title = 'DeliveryETA API', description = des, version = MODEL_VERSION)


# ==========================================================
# 3. ROOT ENDPOINT
# ==========================================================

@app.get('/')  # Human Readable
def home():
    return {
        'message': 'Welcome to DeliveryETA API',
        'model_version': MODEL_VERSION,
        'prediction_endpoint': '/predict',
        'health_endpoint': '/health'
    }

# ==========================================================
# 4. HEALTH CHECK ENDPOINT
# ==========================================================
@app.get('/health')
def health_check():

    if model is None:
        raise HTTPException(status_code = 503, detail = 'Machine Learning model is unavailable')

    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': True
    }

# ==========================================================
# 5. DELIVERY TIME PREDICTION ENDPOINT
# ==========================================================

@app.post('/predict')
def predict_order_delivery_time(new_order: Order_Details):
    
    # Predict the estimated delivery duration for a food order.
    # Workflow:
    #     Step 1: Receive delivery information as JSON.
    #     Step 2: Validate the input using the Order_Details Pydantic schema.
    #     Step 3: Convert the validated Pydantic object into a Python dictionary.
    #     Step 4: Pass this dictionary to the ML inference function.
    #     Step 5: Receive the predicted delivery duration.
    #     Step 6: Return the prediction in JSON format.

    # Args:
    #     new_order (Order_Details):
    #         A Pydantic object containing the delivery-related
    #         features required by the trained ML model.

    # Returns:
    #     dict:
    #         A JSON-compatible dictionary containing the
    #         predicted delivery time in minutes.
    #
    #     Ex: {"model_prediction": 25}

    # ------------------------------------------------------
    # STEP 1: CHECK MODEL AVAILABILITY
    # ------------------------------------------------------

    if model is None:
        raise HTTPException(status_code = 503, detail = 'Machine Learning model is unavailable')

    # ------------------------------------------------------
    # STEP 2: CONVERT PYDANTIC OBJECT INTO A DICTIONARY
    # ------------------------------------------------------
    new_order_dict = new_order.model_dump()

    # ------------------------------------------------------
    # STEP 3: GENERATE PREDICTION USING TRAINED MODEL
    # ------------------------------------------------------
    try:
        prediction = predict_delivery_time(new_order_dict)
        
    except Exception:
    # HTTP 500 indicates an unexpected server-side failure
        raise HTTPException(status_code = 500, detail = 'Unable to generate a prediction. Please try again.')

    # ------------------------------------------------------
    # STEP 3: RETURN SUCCESSFUL PREDICTION AS JSON
    # ------------------------------------------------------
    return JSONResponse(status_code = 200, content = {'model_prediction': round(prediction)})