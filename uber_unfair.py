import random
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

## Notes
# cost distance between driver current location, and passenger pickup location + some function of the euclidean distance between the passenger dropoff location and city center
# in practice drivers don't like far drop-off location (perhaps the probability of a passenger generating)
# show that we have considered these factors and for simplicity here is our function
## Suggestion: cost coefficient (not each driver has same cost) (different for each driver)
# also for revenue distribution, take into consideration how many rides they have picked up
# note in the writeup limitations (such as dynamic, with new drivers and passengers being generated)
# 2-3 papers
# the paper online bi-partite matching, each time a driver or rider comes in, each consumer has willingness to pay



# Base class for Agents
class Agent:
    def __init__(self, name, current_location):
        self.name = name
        self.current_location = current_location

# Derived class for Drivers, inheriting from Agent
class Driver(Agent):
    def __init__(self, name, current_location):
        super().__init__(name, current_location)
        self.total_income = 0
        # number of rides drivers have picked up
        self.total_rides = 0
        self.preference_ordering = []
        # the passenger they picked up in the previous round
        self.matched_passenger_name = None
        # the locations they have visited
        self.locations = []
    
    def cost_to_driver(self, passenger, city_center):
        mean = 1
        sd = 0.2
        cost_coef = random.gauss(mean, sd)
        bare_cost = (abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] 
        - self.current_location[1]) + 0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + 
        (passenger.dropoff_location[1] - city_center[1])**2))**2)
        return cost_coef * bare_cost


    # Calculate preference ordering for passengers
    def calculate_preference_ordering(self, passengers, city_center):
        self.preference_ordering = sorted(passengers, key=lambda passenger: -(passenger.WTP - 
        (abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] 
        - self.current_location[1]) + 0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + 
        (passenger.dropoff_location[1] - city_center[1])**2))**2)))

# Derived class for Passengers, inheriting from Agent
class Passenger(Agent):
    def __init__(self, name, current_location, dropoff_location):
        super().__init__(name, current_location)
        self.dropoff_location = dropoff_location
        self.WTP = self.calculate_WTP()
        self.preference_ordering = []
    
    # Method to calculate Willingness To Pay for passengers
    def calculate_WTP(self):
        euclidean_distance = math.sqrt((self.dropoff_location[0] - 0)**2 + (self.dropoff_location[1] - 0)**2)
        manhattan_distance = abs(self.current_location[0] - self.dropoff_location[0]) + abs(self.current_location[1] - self.dropoff_location[1])
        return euclidean_distance + manhattan_distance

    # Calculate preference ordering for drivers
    def calculate_preference_ordering(self, drivers):
        self.preference_ordering = sorted(drivers, key=lambda driver: -abs(driver.current_location[0] - self.current_location[0]) - abs(driver.current_location[1] - self.current_location[1]))


# Function to generate a list of drivers
def generate_drivers(num_drivers):
    drivers = []
    for i in range(1, num_drivers + 1):
        # Generate random coordinates for current location
        current_location = generate_coordinate_point()
        drivers.append(Driver(f"Driver_{i}", current_location))
    return drivers

# Function to generate a list of passengers
def generate_passengers(num_passengers):
    passengers = []
    for i in range(1, num_passengers + 1):
        # Generate random coordinates for current location and dropoff location for passengers
        current_location = generate_coordinate_point()
        dropoff_location = generate_coordinate_point()
        passengers.append(Passenger(f"Passenger_{i}", current_location, dropoff_location))
    return passengers

# Function to generate a coordinate point with bias towards the center
def generate_coordinate_point():
    mean = 0
    std_deviation = 20  # Adjust the standard deviation as needed

    # Generate coordinates from a normal distribution
    x = int(random.gauss(mean, std_deviation))
    y = int(random.gauss(mean, std_deviation))

    # Ensure the generated coordinates are within the specified range
    x = max(min(x, 50), -50)
    y = max(min(y, 50), -50)

    return x, y

def random_matches(drivers, passengers, city_center):
    random.shuffle(drivers)
    random.shuffle(passengers)

    matches = []

    for i, passenger in enumerate(passengers):
        driver = drivers[i % len(drivers)]
        if driver not in matches:
            matches.append(driver, passenger)

    return matches

def closets_matches(drivers, passengers):
    # Calculate the Euclidean distance matrix between points in drivers and passengers
    driver_locations = [driver.current_location for driver in drivers]
    passenger_locations = [passenger.current_location for passenger in passengers]

    distance_matrix = cdist(driver_locations, passenger_locations)

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create a dictionary to store the matches
    matches = []

    # Populate the matches dictionary
    matches = [(drivers[a].tolist(), passengers[b].tolist()) for a, b in zip(row_ind, col_ind)]

    return matches

    


# Function for driver-proposing deferred acceptance matching
def driver_proposing_matching(drivers, passengers, city_center):
    matches = []
    unmatched_passengers = passengers.copy()
    unmatched_drivers = drivers.copy()

    # Continue proposing until all drivers are matched or no more passengers are available
    while unmatched_drivers and unmatched_passengers:
        for driver in unmatched_drivers:
            # Calculate preference ordering for the current round
            driver.calculate_preference_ordering(unmatched_passengers, city_center)

            # Try to propose to the most preferred passenger
            if driver.preference_ordering:
                best_passenger = driver.preference_ordering[0]

                # Check if the best passenger is unmatched
                if best_passenger in unmatched_passengers:
                    matches.append((driver.name, best_passenger.name))
                    unmatched_passengers.remove(best_passenger)
                    unmatched_drivers.remove(driver)
                    driver.total_income += best_passenger.WTP
                    driver.matched_passenger_name = best_passenger.name
                    break  # Break out of the loop to the next driver

    return matches

# Function to simulate a round of the described simulation
def simulate_round(drivers, passengers, round_num, city_center):
    print(f"\nRound {round_num}:")

    new_passengers = generate_passengers(random.randint(5, 25))
    new_matches = driver_proposing_matching(drivers, new_passengers, city_center)

    # Print the round number and new matches
    print(f"Matches: {new_matches}")

    for driver in drivers:
        if driver.matched_passenger_name is not None:
            # Find the matched passenger in the new_passengers list
            matched_passenger = next((p for p in new_passengers if p.name == driver.matched_passenger_name), None)

            # Check if matched_passenger is not None before accessing attributes
            if matched_passenger is not None:
                driver.current_location = matched_passenger.dropoff_location

    return new_passengers, new_matches

# Main function to initialize and run the simulation
def main():
    round_num = 1
    city_center = (0, 0)

    # Generate drivers only in round 1
    drivers = generate_drivers(15)

    # Generate passengers for round 1
    passengers = generate_passengers(random.randint(5, 25))

    matches = driver_proposing_matching(drivers, passengers, city_center)

    for match in matches:
        driver = next(d for d in drivers if d.name == match[0])
        passenger = next(p for p in passengers if p.name == match[1])
        # Update driver's attributes based on the matched passenger
        driver.total_income += passenger.WTP
        driver.matched_passenger_name = passenger.name

    for round_num in range(2, 51):
        # Generate new passengers for each round
        passengers = generate_passengers(random.randint(5, 25))
        
        # Simulate the round
        new_passengers, matches = simulate_round(drivers, passengers, round_num, city_center)

    # Print total_income values for each driver at the end of the simulation
    print("\nTotal Income of Each Driver:")
    for driver in drivers:
        print(f"{driver.name}: {driver.total_income}")

    print("\nSimulation completed!")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
