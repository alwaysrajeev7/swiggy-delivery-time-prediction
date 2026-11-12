'''Inference-only features matching the original Swiggy datetime rules.'''

from datetime import date, datetime, time, timedelta
from math import cos, pi, sin

import pandas as pd


MODEL_COLUMNS = [
    'rider_age', 'rider_ratings', 'multiple_deliveries', 'vehicle_type',
    'vehicle_condition', 'distance_km', 'order_type',
    'order_pickup_time_mins', 'order_placed_hr', 'sin_hr', 'cos_hr',
    'order_time_of_day', 'is_lunch_rush', 'is_dinner_rush', 'order_day',
    'order_day_of_week', 'is_friday_night', 'is_weekend', 'order_month',
    'is_month_start', 'is_month_end', 'is_festival', 'weather', 'traffic'
]


def build_date_and_time_related_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    '''Transform raw Streamlit values into the model's 24 input columns.

    Expects order_date, order_time, order_picked_time and the other 11
    non-temporal model inputs. Pickup before order on the same date is
    interpreted as crossing midnight, as in the training notebook.
    '''
    df = raw_df.copy()
    if len(df) != 1:
        raise ValueError('Exactly one delivery record is required.')

    row = df.iloc[0]
    order_date = row['order_date']
    order_time = row['order_time']
    picked_time = row['order_picked_time']

    if not isinstance(order_date, date) or isinstance(order_date, datetime):
        raise ValueError('Order date must be a valid date.')
    if not isinstance(order_time, time) or not isinstance(picked_time, time):
        raise ValueError('Order and pickup times must be valid clock times.')

    placed = datetime.combine(order_date, order_time)
    picked = datetime.combine(order_date, picked_time)
    if picked < placed:
        picked += timedelta(days=1)

    minutes = (picked - placed).total_seconds() / 60
    # Historical model supports short pickup durations; reject typos
    # such as choosing an earlier time unintentionally.
    if minutes < 4 or minutes > 15:
        raise ValueError(
            'Pickup must be 4 to 15 minutes after order placement. '
            'If it crosses midnight, choose the next clock time.'
        )

    hour = placed.hour
    angle = 2 * pi * hour / 24
    if 5 <= hour < 12:
        time_of_day = 'Morning'
    elif 12 <= hour < 17:
        time_of_day = 'Afternoon'
    elif 17 <= hour < 21:
        time_of_day = 'Evening'
    elif 21 <= hour < 24:
        time_of_day = 'Night'
    else:
        time_of_day = 'Late Night'

    weekday = placed.strftime('%A')
    next_date = placed + timedelta(days=1)
    values = {
        'rider_age': int(row['rider_age']),
        'rider_ratings': float(row['rider_ratings']),
        'multiple_deliveries': int(row['multiple_deliveries']),
        'vehicle_type': str(row['vehicle_type']),
        'vehicle_condition': int(row['vehicle_condition']),
        'distance_km': float(row['distance_km']),
        'order_type': str(row['order_type']),
        'order_pickup_time_mins': int(minutes) if minutes.is_integer() else minutes,
        'order_placed_hr': hour,
        'sin_hr': round(sin(angle), 4),
        'cos_hr': round(cos(angle), 4),
        'order_time_of_day': time_of_day,
        'is_lunch_rush': int(13 <= hour < 16),
        'is_dinner_rush': int(20 <= hour < 23),
        'order_day': placed.day,
        'order_day_of_week': weekday,
        'is_friday_night': int(weekday == 'Friday' and hour >= 18),
        'is_weekend': int(weekday in ('Saturday', 'Sunday')),
        'order_month': placed.month,
        'is_month_start': int(placed.day == 1),
        'is_month_end': int(next_date.month != placed.month),
        'is_festival': int(row['is_festival']),
        'weather': str(row['weather']),
        'traffic': str(row['traffic'])
    }
    return pd.DataFrame([values], columns=MODEL_COLUMNS)


def prepare_prediction_json(raw_df: pd.DataFrame) -> dict:
    '''Create a JSON-native 24-feature payload plus calendar audit metadata.'''
    engineered = build_date_and_time_related_features(raw_df)
    payload = engineered.iloc[0].to_dict()
    row = raw_df.iloc[0]
    payload['order_datetime'] = datetime.combine(
        row['order_date'], row['order_time']
    ).isoformat()
    return payload
