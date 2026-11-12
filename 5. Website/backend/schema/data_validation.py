from pydantic import BaseModel, Field
from typing import Annotated, Literal

class Order_Details(BaseModel):

    # Rider related features
    rider_age: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 20, le = 39, description = 'Rider age in years', examples = [36])]
    rider_ratings: Annotated[float, Field(..., strict = True, allow_inf_nan = False, ge = 2.5, le = 5.0, description = 'Rider rating', examples = [4.5])]
    multiple_deliveries: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 3, description = 'Number of multiple deliveries rider has to complete', examples = [0])]
    vehicle_type: Annotated[Literal['Motor-Cycle', 'Scooter', 'Electric-Scooter', 'Bicycle'], Field(..., description = 'Rider Delivery vehicle type', examples = ['Motor-Cycle'])]
    vehicle_condition: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 3, description = 'Rider Vehicle condition code', examples = [0])]
    distance_km: Annotated[float, Field(..., strict = True, allow_inf_nan = False, gt = 0, le = 21, description = 'Haversine distance in kilometres that rider have to covered', examples = [4.2])]
    
    # Ordert & Order hour related features
    order_type: Annotated[Literal['Snack', 'Meal', 'Drinks', 'Buffet'], Field(..., description = 'Food order category', examples = ['Meal'])]
    order_placed_hr: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 23, description = 'Order placement hour in 24-hour format', examples = [19])]
    sin_hr: Annotated[float, Field(..., strict = True, allow_inf_nan = False, ge = -1, le = 1, description = 'Sine encoding of order placement hour', examples = [-0.9659258262890684])]
    cos_hr: Annotated[float, Field(..., strict = True, allow_inf_nan = False, ge = -1, le = 1, description = 'Cosine encoding of order placement hour', examples = [0.2588190451025203])]
    order_pickup_time_mins: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 4, le = 15, description = 'Order pickup duration in minutes', examples = [5])]
    
    # Order time category related features
    order_time_of_day: Annotated[Literal['Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'], Field(..., description = 'Time of day category', examples = ['Evening'])]
    is_lunch_rush: Annotated[int, Field(...,  strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Lunch rush indicator',  examples = [0])]
    is_dinner_rush: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Dinner rush indicator', examples = [1])]
    
    # Order day related features
    order_day: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 1, le = 31, description = 'Day of the month', examples = [16])]
    order_day_of_week: Annotated[Literal['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], Field(..., description = 'Day of the week', examples = ['Thursday'])]
    is_friday_night: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Friday night indicator', examples = [0])]
    is_weekend: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Weekend indicator', examples = [1])]

    # Order month related features
    order_month: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 1, le = 12, description = 'Order month number', examples = [3])]
    is_month_start: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Month start indicator', examples = [0])]
    is_month_end: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Month end indicator', examples = [0])]

    # Other features
    is_festival: Annotated[int, Field(..., strict = True, allow_inf_nan = False, ge = 0, le = 1, description = 'Festival indicator', examples = [1])]
    weather: Annotated[Literal['Fog', 'Stormy', 'Cloudy', 'Sandstorms', 'Windy', 'Sunny'], Field(..., description = 'Weather condition', examples = ['Fog'])]
    traffic: Annotated[Literal['Low', 'Medium', 'High', 'Jam'], Field(..., description = 'Traffic density category', examples = ['Medium'])]