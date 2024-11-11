from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    redirect,
    url_for,
)
import os
import requests
import pandas as pd

app = Flask(__name__)

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/formula")
def formula():
    return render_template('formula.html')

@app.route("/driving-dashboard")
def driving_dashboard():
    vehicle_id = request.args.get("car_number")
    if not vehicle_id:
        return jsonify({"error": "Vehicle ID is required"}), 400

    # Read the CSV file
    df = pd.read_csv("driving.csv")

    # Check if vehicle exists
    if vehicle_id not in df["car_number"].unique():
        return jsonify({"error": f"Vehicle {vehicle_id} not found"}), 404

    # If vehicle exists, filter the data for that specific vehicle
    vehicle_data = df[df["car_number"] == vehicle_id]
    car_number = request.args.get("car_number")

    # Prepare the data for charts
    chart_data = {
        "tripId": vehicle_data["TripCount"].tolist(),
        "fuelConsumed": vehicle_data["FuelConsumed"].tolist(),
        "speed": vehicle_data["Speed"].tolist(),
        "engineRPM": vehicle_data["EngineRPM"].tolist(),
        "angularVelocity": vehicle_data["AngularVelocity"].tolist(),
        "brakePedalPressure": vehicle_data["BrakePedalPressure"].tolist(),
        "steeringAngle": vehicle_data["SteeringAngle_A"].tolist(),
        "overallDrivingScore": vehicle_data["OverallDrivingScore"].tolist(),
    }

    return render_template(
        "driving-dashboard.html", car_number=vehicle_id, chart_data=chart_data
    )


@app.route("/min")
def mmin():
    return render_template("min.html")


@app.route("/max")
def mmmax():
    return render_template("max.html")


@app.route("/avg")
def avg():
    return render_template("avg.html")


@app.route("/trip-details")
def trip_dashboard():
    trip_id = request.args.get("tripId")
    df = pd.read_csv("driving.csv")
    selected_trip_data = df[df["TripCount"] == int(trip_id)].iloc[0]

    trip_data = {
        "TripCount": selected_trip_data["TripCount"],
        "Speed": selected_trip_data["Speed"],
        "FuelConsumed": selected_trip_data["FuelConsumed"],
        "EngineRPM": selected_trip_data["EngineRPM"],
        "BrakePedalPressure": selected_trip_data["BrakePedalPressure"],
        "AngularVelocity": selected_trip_data["AngularVelocity"],
        "SteeringAngle_A": selected_trip_data["SteeringAngle_A"],
        "OverallDrivingScore": selected_trip_data["OverallDrivingScore"],
    }
    return render_template("trip-details.html", trip_id=trip_id, trip_data=trip_data)


# @app.route("/trip-details/minimum")
# def car_minimum():
#     return render_template("minimum.html")


# @app.route("/trip-details/maximum")
# def car_maximum():
#     return render_template("maximum.html")


@app.route("/driving.csv")
def serve_csv():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)), "driving.csv"
    )


@app.route("/car-login")
def car_login():
    return render_template("car-login.html")


@app.route("/car-details")
def car_detail():
    return render_template("car-details.html")


def calculate_idv(initial_car_price, vehicle_age):
    """Calculate Insured Declared Value (IDV) based on initial price and vehicle age."""
    # Define depreciation percentage based on vehicle age
    if vehicle_age <= 0.5:
        depreciation_percentage = 0.05
    elif vehicle_age <= 1:
        depreciation_percentage = 0.15
    elif vehicle_age <= 2:
        depreciation_percentage = 0.20
    elif vehicle_age <= 3:
        depreciation_percentage = 0.30
    elif vehicle_age <= 4:
        depreciation_percentage = 0.40
    elif vehicle_age <= 5:
        depreciation_percentage = 0.50
    else:
        depreciation_percentage = 0.60  # Higher depreciation after 5 years

    # Calculate IDV as the initial price minus depreciation
    idv = initial_car_price * (1 - depreciation_percentage)
    return idv


def calculate_car_valuation(
    initial_car_price,
    vehicle_age,
    condition_score,
    market_demand_index,
    seller_type_factor,
    brand_reputation_factor,
    fuel_type,
    num_insurance_claims,
    km_driven,
):
    """Calculate the car valuation considering various factors."""
    # Calculate IDV
    idv = calculate_idv(initial_car_price, vehicle_age)

    # Define expected lifespan based on fuel type
    expected_lifespan = 10 if fuel_type.lower() == "petrol" else 15

    # Calculate depreciation factor based on vehicle age and lifespan
    depreciation_factor = (
        min(vehicle_age / expected_lifespan, 0.2)
        if vehicle_age <= 1
        else vehicle_age / expected_lifespan
    )
    depreciation_factor = min(depreciation_factor, 0.9)  # Cap depreciation to 90%

    # Calculate insurance claim depreciation
    total_insurance_depreciation = min(
        0.05 * num_insurance_claims, 0.5
    )  # 5% reduction per claim, capped at 50%

    # Kilometers driven impact (annual limit: 8,000 km)
    avg_km_per_year = vehicle_age
    km_depreciation_factor = (
        0.1 if avg_km_per_year > 8000 else 0.0
    )  # Additional 10% depreciation if over limit

    # Adjust insurance impact for older cars with fewer claims
    if vehicle_age >= 5 and num_insurance_claims < 2 and km_driven > 40000:
        total_insurance_depreciation *= 0.5  # Reduce insurance impact by 50%

    # Condition score and market adjustment
    condition_factor = condition_score
    market_trend_adjustment = 1 + ((market_demand_index - 1) / 10)

    # Ensure seller type factor is within valid range
    seller_type_factor = max(0.6, min(seller_type_factor, 1.0))

    # Calculate final car valuation
    car_valuation = (
        initial_car_price
        * (1 - depreciation_factor - km_depreciation_factor)
        * (1 - total_insurance_depreciation)
        * condition_factor
        * market_trend_adjustment
        * seller_type_factor
        * brand_reputation_factor
    )

    # Ensure car valuation is at least 10% higher than IDV and not below 50,000
    car_valuation = max(
        car_valuation, idv * 1.10
    )  # Favor buyers with at least 10% above IDV
    car_valuation = max(car_valuation, 50000)  # Ensure minimum valuation of 50,000

    return car_valuation, idv
@app.route("/calculate-manual-valuation", methods=["POST"])
def calculate_manual_valuation():
    # Retrieve data from form submission
    initial_car_price = float(request.form.get("initial_car_price"))
    vehicle_age = int(request.form.get("vehicle_age"))
    condition_score = float(request.form.get("condition_score"))
    market_demand_index = float(request.form.get("market_demand_index"))
    seller_type_factor = float(request.form.get("seller_type_factor"))
    brand_reputation_factor = float(request.form.get("brand_reputation_factor"))
    fuel_type = request.form.get("fuel_type")
    num_insurance_claims = int(request.form.get("num_insurance_claims"))
    km_driven = int(request.form.get("km_driven"))

    # Perform the calculation using the same function
    car_valuation, idv = calculate_car_valuation(
        initial_car_price,
        vehicle_age,
        condition_score,
        market_demand_index,
        seller_type_factor,
        brand_reputation_factor,
        fuel_type,
        num_insurance_claims,
        km_driven,
    )

    # Return the result back to the HTML template
    return render_template("index.html", car_valuation=car_valuation, idv=idv)



@app.route("/trip-details", methods=["GET"])
def get_trip_details():
    try:
        # Get parameters from the URL
        trip_id = request.args.get("tripId")
        car_number = request.args.get("car_number")

        # Validate parameters
        if not trip_id or not car_number:
            return (
                jsonify(
                    {
                        "error": "Missing required parameters. Both tripId and carNumber are required."
                    }
                ),
                400,
            )

        # Load the data
        df = load_csv_data()
        if df is None:
            return jsonify({"error": "Error loading data"}), 500

        # Find the matching row
        # Convert trip_id to the appropriate type if needed (e.g., to int)
        trip_id = int(trip_id)
        matching_row = df[(df["trip_id"] == trip_id) & (df["car_number"] == car_number)]

        if matching_row.empty:
            return (
                jsonify(
                    {"error": "No data found for the specified tripId and carNumber"}
                ),
                404,
            )

        # Convert the row to a dictionary
        trip_data = matching_row.iloc[0].to_dict()

        # Format the response
        response = {
            "trip_info": {
                "car_number": trip_data["car_number"],
                "trip_id": trip_data["trip_id"],
                "TripCount": trip_data.get("trip_count", None),
                "FuelConsumed": trip_data.get("fuel_consumed", None),
                "Speed": trip_data.get("speed", None),
                "EngineRPM": trip_data.get("engine_rpm", None),
                "AngularVelocity": trip_data.get("angular_velocity", None),
                "BrakePedalPressure": trip_data.get("brake_pedal_pressure", None),
                "SteeringAngle_A": trip_data.get("steering_angle", None),
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/submit-car-number", methods=["POST"])
def submit_car_number():
    car_number = request.form["car_number"]
    # km_driven = request.form["km_driven"]

    # Read the vehicle_info.csv file to get car details
    vehicle_info = pd.read_csv("vehicle_info.csv")

    # Find the car details based on the submitted car number
    car_data = vehicle_info[vehicle_info["registration_number"] == car_number]

    # If car data is found, perform the valuation
    if not car_data.empty:
        # Extract required fields from the CSV for the car
        car_data = car_data.iloc[0]
        ex_showroom_price = car_data["ex_showroom_price"]
        if ex_showroom_price == 0:
            return f"Invalid ex-showroom price for car number {car_number}. Please check the details."

        vehicle_age = 2024 - int(car_data["year"])
        fuel_type = car_data["fuel_descr"]
        num_insurance_claims = int(car_data["insurance_claims"])

        # Calculate car valuation
        predicted_price, idv = calculate_car_valuation(
            initial_car_price=ex_showroom_price,
            vehicle_age=vehicle_age,
            condition_score=0.9,  # Assuming a default condition score
            market_demand_index=1.0,  # Assuming normal market demand
            seller_type_factor=0.8,  # Adjust for seller type
            brand_reputation_factor=0.95,  # Adjust for brand reputation
            fuel_type=fuel_type,
            num_insurance_claims=num_insurance_claims,
            km_driven=1000
        )

        # Pass the car_number as a hidden field or URL parameter in car-details.html
        return render_template(
            "car-details.html",
            car_data=car_data,
            predicted_price=predicted_price,
            idv=idv,
            car_number=car_number,  # Added this to pass to template
        )
    else:
        return f"Car number {car_number} not found in the system. Please check the car number and try again."


if __name__ == "__main__":
    app.run(debug=True)