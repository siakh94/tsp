import os
import json
import random

# Define base directory
base_dir = "data"
station_set = {"Seattle": "RouteID_1", "New_York": "RouteID_2", "Los_Angeles": "RouteID_3"}

# Real-world locations for each city
city_locations = {
    "Seattle": ["Pike Place Market", "Space Needle", "Seattle Art Museum", "University of Washington", "Alki Beach"],
    "New_York": ["Times Square", "Central Park", "Brooklyn Bridge", "Empire State Building", "Statue of Liberty"],
    "Los_Angeles": ["Hollywood Sign", "Venice Beach", "Santa Monica Pier", "Griffith Observatory", "LAX Airport"]
}


# Generate realistic travel times per city
def generate_travel_times(city):
    stops = city_locations[city]
    travel_times = {}
    for i in range(len(stops)):
        travel_times[stops[i]] = {}
        for j in range(len(stops)):
            if i != j:
                travel_times[stops[i]][stops[j]] = random.randint(5, 45)  # Simulating realistic urban travel times
    return travel_times


# Generate realistic package data per city
def generate_package_data(city):
    package_data = {
        station_set[city]: {"packages": {f"PKG{j}": random.choice(city_locations[city]) for j in range(1, 4)}}}
    return package_data


# Generate AI-predicted sequences (proposed routes)
def generate_proposed_sequences(city):
    proposed_sequences = {
        station_set[city]: {"proposed": random.sample(city_locations[city], len(city_locations[city]))}}
    return proposed_sequences


# Generate actual sequences that match the city
def generate_actual_sequences(city):
    actual_sequences = {station_set[city]: {"actual": random.sample(city_locations[city], len(city_locations[city]))}}
    return actual_sequences


# Sample route metadata
sample_route_data = {
    "RouteID_1": {"route_score": random.choice(["High", "Medium", "Low"]), "station_code": "Seattle"},
    "RouteID_2": {"route_score": random.choice(["High", "Medium", "Low"]), "station_code": "New_York"},
    "RouteID_3": {"route_score": random.choice(["High", "Medium", "Low"]), "station_code": "Los_Angeles"},
}

sample_invalid_sequence_scores = {"RouteID_3": random.randint(500, 1500)}

sample_model_output = {
    "best_params": {"t1": random.uniform(1.0, 10.0), "t2": random.uniform(1.0, 10.0), "t3": random.uniform(1.0, 10.0),
                    "t4": random.uniform(1.0, 10.0), "t5": random.uniform(1.0, 10.0)}
}


# Create necessary directories
def create_directories():
    for city in station_set:
        os.makedirs(f"{base_dir}/model_build_inputs_{city}", exist_ok=True)
    os.makedirs(f"{base_dir}/model_apply_inputs", exist_ok=True)
    os.makedirs(f"{base_dir}/model_score_inputs", exist_ok=True)
    os.makedirs(f"{base_dir}/model_build_outputs", exist_ok=True)


# Save JSON files
def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# Generate files
def generate_files():
    create_directories()

    for city in station_set:
        build_dir = f"{base_dir}/model_build_inputs_{city}"
        save_json(f"{build_dir}/route_data.json", sample_route_data)
        save_json(f"{build_dir}/package_data.json", generate_package_data(city))
        save_json(f"{build_dir}/proposed_sequences.json", generate_proposed_sequences(city))
        save_json(f"{build_dir}/actual_sequences.json", generate_actual_sequences(city))
        save_json(f"{build_dir}/travel_times.json", generate_travel_times(city))
        save_json(f"{build_dir}/invalid_sequence_scores.json", sample_invalid_sequence_scores)

    save_json(f"{base_dir}/model_apply_inputs/new_route_data.json", sample_route_data)
    save_json(f"{base_dir}/model_apply_inputs/new_package_data.json", generate_package_data("Seattle"))
    save_json(f"{base_dir}/model_apply_inputs/new_travel_times.json", generate_travel_times("Seattle"))

    save_json(f"{base_dir}/model_score_inputs/new_proposed_sequences.json", generate_proposed_sequences("Seattle"))
    save_json(f"{base_dir}/model_score_inputs/new_actual_sequences.json", generate_actual_sequences("Seattle"))
    save_json(f"{base_dir}/model_score_inputs/new_invalid_sequence_scores.json", sample_invalid_sequence_scores)

    for city in station_set:
        save_json(f"{base_dir}/model_build_outputs/model_{city}.json", sample_model_output)


if __name__ == "__main__":
    generate_files()
    print("✅ JSON files with AI-generated proposed routes and real-world actual routes have been generated!")
