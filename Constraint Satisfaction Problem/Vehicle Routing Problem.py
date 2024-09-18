
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Constants
NUM_VEHICLES = 2
LOCATIONS = {
    'Depot': (0, 0),
    'Location 1': (1, 2),
    'Location 2': (2, 4),
    'Location 3': (3, 1),
    'Location 4': (5, 3)
}

# Calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Total distance for a given route
def total_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(LOCATIONS[route[i]], LOCATIONS[route[i + 1]])
    return distance

# Generate all possible routes for the vehicles
def generate_routes(locations):
    loc_names = list(locations.keys())
    return list(permutations(loc_names[1:]))  # Exclude the depot for permutations

# Find the best routes for the vehicles
def find_best_routes(locations):
    all_routes = generate_routes(locations)
    best_routes = None
    best_distance = float('inf')

    for route in all_routes:
        # Split route into segments for vehicles
        vehicle_routes = np.array_split(route, NUM_VEHICLES)
        total_dist = 0
        
        for vehicle_route in vehicle_routes:
            full_route = ['Depot'] + list(vehicle_route) + ['Depot']
            total_dist += total_distance(full_route)
        
        if total_dist < best_distance:
            best_distance = total_dist
            best_routes = vehicle_routes
            
    return best_routes, best_distance

# Visualize the routes
def visualize_routes(routes):
    plt.figure(figsize=(8, 8))
    for i, vehicle_route in enumerate(routes):
        route_with_depot = ['Depot'] + list(vehicle_route) + ['Depot']
        for j in range(len(route_with_depot) - 1):
            start = LOCATIONS[route_with_depot[j]]
            end = LOCATIONS[route_with_depot[j + 1]]
            plt.plot([start[0], end[0]], [start[1], end[1]], marker='o', label=f'Vehicle {i+1}' if j == 0 else "")
    
    # Mark locations
    for location, coords in LOCATIONS.items():
        plt.text(coords[0], coords[1], location, fontsize=12, ha='right')
    
    plt.title("Vehicle Routing Problem Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.legend()
    plt.show()

# Main function to run the VRP game
def run_vrp_game():
    best_routes, best_distance = find_best_routes(LOCATIONS)
    print("Best Routes:")
    for i, vehicle_route in enumerate(best_routes):
        print(f"Vehicle {i+1}: {' -> '.join(vehicle_route)}")
    print(f"Total Distance: {best_distance:.2f}")
    
    visualize_routes(best_routes)

# Run the Vehicle Routing Problem game
run_vrp_game()
