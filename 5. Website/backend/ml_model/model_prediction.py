
"""
DeliveryETA — Machine Learning Model Inference

Responsibilities:
1. Load the saved Gradient Boosting model pipeline.
2. Validate the expected input feature structure.
3. Convert input data into a pandas DataFrame.
4. Generate the delivery time prediction.
5. Handle model loading and prediction errors.
"""

import math
import pickle
import pandas as pd
from pathlib import Path

# ==========================================================
# 1. MODEL CONFIGURATION
# ==========================================================

MODEL_VERSION = "1.0.0"

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "gbr_model_version_1.pkl"
)

# ==========================================================
# 2. FEATURES EXPECTED BY THE TRAINED MODEL
# ==========================================================
FEATURE_COLUMNS = ['rider_age', 'rider_ratings', 'multiple_deliveries', 'vehicle_type', 'vehicle_condition', 'distance_km', 'order_type', 
                   'order_pickup_time_mins', 'order_placed_hr', 'sin_hr', 'cos_hr', 'order_time_of_day', 'is_lunch_rush', 'is_dinner_rush',
                   'order_day', 'order_day_of_week', 'is_friday_night', 'is_weekend', 'order_month', 'is_month_start', 'is_month_end',
                   'is_festival', 'weather', 'traffic']


# ==========================================================
# 3. LOAD THE TRAINED MACHINE LEARNING MODEL
# ==========================================================

model = None

try:
   with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

except Exception as e:
    model = None
    print(f"Model loading failed: {type(e).__name__}: {e}", flush = True)

# ==========================================================
# 4. GENERATE DELIVERY TIME PREDICTION
# ==========================================================

def predict_delivery_time(user_input: dict) -> float:
  
   # ------------------------------------------------------
   # STEP 1: CHECK MODEL AVAILABILITY
   # ------------------------------------------------------
   if model is None:
      raise RuntimeError('ML model is not loaded. Prediction unavailable.')

   # ------------------------------------------------------
   # STEP 2: VALIDATE INPUT STRUCTURE
   # ------------------------------------------------------
   if not isinstance(user_input, dict):
      raise ValueError('Input must be a Python dictionary.')

   missing_features = [feature for feature in FEATURE_COLUMNS if feature not in user_input]
   if len(missing_features) > 0:
      raise ValueError(f'feature: {missing_features} is missing')

   # ------------------------------------------------------
   # STEP 3: CONVERT INPUT INTO A DATAFRAME
   # ------------------------------------------------------
   input_df = pd.DataFrame(data = [user_input], columns = FEATURE_COLUMNS)

   # ------------------------------------------------------
   # STEP 4: GENERATE PREDICTION
   # ------------------------------------------------------
   prediction = model.predict(input_df)[0]
   
   predicted_minutes = float(prediction)

   # Ensure the model returned a valid number.
   if not math.isfinite(predicted_minutes):
      raise ValueError('Model returned a non-finite prediction.')

   # ------------------------------------------------------
   # STEP 5: RETURN PREDICTION
   # ------------------------------------------------------
   return predicted_minutes