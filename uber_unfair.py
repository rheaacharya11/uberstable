import random
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import copy
import statistics
import matplotlib.pyplot as plt

## Notes
# cost distance between driver current location, and passenger pickup location + some function of the euclidean distance between the passenger dropoff location and city center
# in practice drivers don't like far drop-off location (perhaps the probability of a passenger generating)
# show that we have considered these factors and for simplicity here is our function
## Suggestion: cost coefficient (not each driver has same cost) (different for each driver)
# also for revenue distribution, take into consideration how many rides they have picked up
# note in the writeup limitations (such as dynamic, with new drivers and passengers being generated)
# 2-3 papers
# the paper online bi-partite matching, each time a driver or rider comes in, each consumer has willingness to pay

# NUM_AGENTS = 15

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
    def calculate_preference_ordering_driver_new(self, passengers, city_center):
        p = 0.9
        self.preference_ordering = sorted(passengers, key=lambda passenger: -(passenger.WTP - 
        2*((1-p)*(abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] 
        - self.current_location[1])) + p*0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + 
        (passenger.dropoff_location[1] - city_center[1])**2))**2)))
    
    def calculate_preference_ordering_driver(self, passengers, city_center):
        p = 0.1
        self.preference_ordering = sorted(passengers, key=lambda passenger: -(passenger.WTP - 
        2*((1-p)*(abs(passenger.current_location[0] - self.current_location[0]) + abs(passenger.current_location[1] 
        - self.current_location[1])) + p*0.1 * (math.sqrt((passenger.dropoff_location[0] - city_center[0])**2 + 
        (passenger.dropoff_location[1] - city_center[1])**2))**2)))


# Derived class for Passengers, inheriting from Agent
class Passenger(Agent):
    def __init__(self, name, current_location, dropoff_location):
        super().__init__(name, current_location)
        self.dropoff_location = dropoff_location
        self.WTP = self.calculate_WTP()
        self.preference_ordering = []

        self.matched_driver_name = None
    
    # Method to calculate Willingness To Pay for passengers
    def calculate_WTP(self):
        mean = 1
        sd = 0.1
        wtp = random.gauss(mean, sd)
        euclidean_distance1 = math.sqrt((self.dropoff_location[0] - 0)**2 + (self.dropoff_location[1] - 0)**2)
        euclidean_distance2 = math.sqrt((self.current_location[0] - 0)**2 + (self.current_location[1] - 0)**2)
        manhattan_distance = abs(self.current_location[0] - self.dropoff_location[0]) + abs(self.current_location[1] - self.dropoff_location[1])
        return wtp*((euclidean_distance1+ euclidean_distance2)/2 + manhattan_distance)/3

    # Calculate preference ordering for drivers
    def calculate_preference_ordering_passenger(self, drivers, var):
        def man_dist(driver):
            return (abs(driver.current_location[0] - self.current_location[0]) 
            + abs(driver.current_location[1] - self.current_location[1]))
        
        driver_dict = {driver: (man_dist(driver), driver.total_income) for driver in drivers}
        values = list(driver_dict.values())

        # Separate the values into two lists for the first and second functions
        values_function1, values_function2 = zip(*values)

        # Normalize the values using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_function1 = scaler.fit_transform([[value] for value in values_function1])
        normalized_function2 = scaler.fit_transform([[value] for value in values_function2])

        # Update the dictionary with normalized values
        normal_driver_dict = {driver: (normalized_function1[i][0] + var*normalized_function2[i][0]) for i, driver in enumerate(driver_dict.keys())}
        self.preference_ordering = sorted(normal_driver_dict.keys(), key=lambda key: normal_driver_dict[key])

def manh_dist(driver, passenger):
            return (abs(driver.current_location[0] - passenger.current_location[0]) 
            + abs(driver.current_location[1] - passenger.current_location[1]))

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

def random_matches(drivers, passengers):
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    random.shuffle(drivers)
    random.shuffle(passengers)

    matches = []

    for i, passenger in enumerate(passengers):
        driver = drivers[i % len(drivers)]
        if driver not in matches:
            matches.append((driver.name, passenger.name))
    
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)

    return matches

def closest_matches(drivers, passengers):
    driver_dict_name = {driver.name: driver for driver in drivers}
    passenger_dict_name = {passenger.name: passenger for passenger in passengers}
    
    # Calculate the Euclidean distance matrix between points in drivers and passengers
    driver_locations = [driver.current_location for driver in drivers]
    passenger_locations = [passenger.current_location for passenger in passengers]


    driver_dict = {driver.current_location: driver for driver in drivers}
    passenger_dict = {passenger.current_location: passenger for passenger in passengers}

    distance_matrix = cdist(driver_locations, passenger_locations)

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create a dictionary to store the matches
    matches = []

    for i, j in zip(row_ind, col_ind):
        driver_point = driver_locations[i]
        passenger_point = passenger_locations[j]
        driver = driver_dict[driver_point] 
        passenger = passenger_dict[passenger_point]
        matches.append((driver.name, passenger.name))

    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict_name[passenger_name].WTP < manh_dist(driver_dict_name[driver_name], passenger_dict_name[passenger_name]):
            matches.remove(match)

    return matches
        




# Function for driver-proposing deferred acceptance matching
def driver_proposing_matching(drivers, passengers, city_center, var):
    matches = []
    unmatched_passengers = passengers.copy()
    unmatched_drivers = drivers.copy()

    # clear out last round
    for i in range(len(drivers)):
        drivers[i].matched_passenger_name = None 
        drivers[i].calculate_preference_ordering_driver(passengers, city_center)
        passengers[i].matched_passenger_name = None
        passengers[i].calculate_preference_ordering_passenger(drivers, var)

    # dicts from names to objects
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}


    passenger_prefs = {passenger.name: {driver.name: i for (i, driver) in enumerate(passenger.preference_ordering)} for passenger in passengers}
    driver_prefs = {driver.name: {passenger.name: i for (i, passenger) in enumerate(driver.preference_ordering)} for driver in drivers}

    # initialize set of unmatched renters
    free_drivers = set(drivers)

    while free_drivers:
        driver = free_drivers.pop()

        for preferred_passenger in driver.preference_ordering:
            if preferred_passenger.matched_driver_name is None:
                preferred_passenger.matched_driver_name = driver.name
                driver.matched_passenger_name = preferred_passenger.name
                break 
            else:
                current_match_rank = passenger_prefs[preferred_passenger.name][preferred_passenger.matched_driver_name]
                new_match_rank = passenger_prefs[preferred_passenger.name][driver.name]

                if new_match_rank < current_match_rank:
                    # If the lender prefers the new renter, make the switch
                    free_drivers.add(driver_dict[preferred_passenger.matched_driver_name])
                    preferred_passenger.matched_driver_name = driver.name
                    driver.matched_passenger_name = preferred_passenger.name
                    # print("matched non-empty")
                    #print(f"non-empty match: {driver.name} matched with {preferred_passenger.name}")
                    break


    matches = [(driver.name, driver.matched_passenger_name) for driver in drivers]
    #filter the matches where wait time too long
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)
    
    

    # # Continue proposing until all drivers are matched or no more passengers are available
    # while unmatched_drivers and unmatched_passengers:
    #     for driver in unmatched_drivers:
    #         # Calculate preference ordering for the current round
    #         driver.calculate_preference_ordering(unmatched_passengers, city_center)

    #         # Try to propose to the most preferred passenger
    #         if driver.preference_ordering:
    #             best_passenger = driver.preference_ordering[0]

    #             # Check if the best passenger is unmatched
    #             if best_passenger in unmatched_passengers:
    #                 matches.append((driver.name, best_passenger.name))
    #                 unmatched_passengers.remove(best_passenger)
    #                 unmatched_drivers.remove(driver)
    #                 driver.total_income += best_passenger.WTP
    #                 driver.matched_passenger_name = best_passenger.name
    #                 break  # Break out of the loop to the next driver

    return matches

def driver_proposing_matching_new(drivers, passengers, city_center, var):
    matches = []
    unmatched_passengers = passengers.copy()
    unmatched_drivers = drivers.copy()

    # clear out last round
    for i in range(len(drivers)):
        drivers[i].matched_passenger_name = None 
        drivers[i].calculate_preference_ordering_driver_new(passengers, city_center)
        passengers[i].matched_passenger_name = None
        passengers[i].calculate_preference_ordering_passenger(drivers, var)

    # dicts from names to objects
    driver_dict = {driver.name: driver for driver in drivers}
    passenger_dict = {passenger.name: passenger for passenger in passengers}


    passenger_prefs = {passenger.name: {driver.name: i for (i, driver) in enumerate(passenger.preference_ordering)} for passenger in passengers}
    driver_prefs = {driver.name: {passenger.name: i for (i, passenger) in enumerate(driver.preference_ordering)} for driver in drivers}

    # initialize set of unmatched renters
    free_drivers = set(drivers)

    while free_drivers:
        driver = free_drivers.pop()

        for preferred_passenger in driver.preference_ordering:
            if preferred_passenger.matched_driver_name is None:
                preferred_passenger.matched_driver_name = driver.name
                driver.matched_passenger_name = preferred_passenger.name
                break 
            else:
                current_match_rank = passenger_prefs[preferred_passenger.name][preferred_passenger.matched_driver_name]
                new_match_rank = passenger_prefs[preferred_passenger.name][driver.name]
                if new_match_rank < current_match_rank:
                    # If the lender prefers the new renter, make the switch
                    free_drivers.add(driver_dict[preferred_passenger.matched_driver_name])
                    preferred_passenger.matched_driver_name = driver.name
                    driver.matched_passenger_name = preferred_passenger.name
                    # print("matched non-empty")
                    #print(f"non-empty match: {driver.name} matched with {preferred_passenger.name}")
                    break


    matches = [(driver.name, driver.matched_passenger_name) for driver in drivers]
    #filter the matches where wait time too long
    for match in matches:
        driver_name, passenger_name = match
        if passenger_dict[passenger_name].WTP < manh_dist(driver_dict[driver_name], passenger_dict[passenger_name]):
            matches.remove(match)
    
    

    # # Continue proposing until all drivers are matched or no more passengers are available
    # while unmatched_drivers and unmatched_passengers:
    #     for driver in unmatched_drivers:
    #         # Calculate preference ordering for the current round
    #         driver.calculate_preference_ordering(unmatched_passengers, city_center)

    #         # Try to propose to the most preferred passenger
    #         if driver.preference_ordering:
    #             best_passenger = driver.preference_ordering[0]

    #             # Check if the best passenger is unmatched
    #             if best_passenger in unmatched_passengers:
    #                 matches.append((driver.name, best_passenger.name))
    #                 unmatched_passengers.remove(best_passenger)
    #                 unmatched_drivers.remove(driver)
    #                 driver.total_income += best_passenger.WTP
    #                 driver.matched_passenger_name = best_passenger.name
    #                 break  # Break out of the loop to the next driver

    return matches

# Function to simulate a round of the described simulation
def simulate_round(drivers, new_passengers, round_num, city_center, algo, var):
    # print(f"\nRound {round_num}:")

    driver_dict = {driver.name: driver for driver in drivers}

    passenger_dict = {passenger.name: passenger for passenger in new_passengers}

    if algo == driver_proposing_matching:
        new_matches = driver_proposing_matching(drivers, new_passengers, city_center, 0)
    if algo == driver_proposing_matching_new:
        new_matches = driver_proposing_matching_new(drivers, new_passengers, city_center, 5)
    elif algo == random_matches:
        new_matches = random_matches(drivers, new_passengers)
    else:
        new_matches = closest_matches(drivers, new_passengers)

    for match in new_matches:
        for driver in drivers:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
        if driver.matched_passenger_name is not None:
            passenger = passenger_dict[driver.matched_passenger_name]
            driver.current_location = passenger.dropoff_location
            driver.total_income += passenger.WTP

        # if driver.matched_passenger_name is not None:
        #     # Find the matched passenger in the new_passengers list
        #     matched_passenger = next((p for p in new_passengers if p.name == driver.matched_passenger_name), None)

        #     # Check if matched_passenger is not None before accessing attributes
        #     if matched_passenger is not None:
        #         driver.current_location = matched_passenger.dropoff_location
        #         driver.total_income += matched_passenger.WTP
    return new_matches, len(new_matches)

# Main function to initialize and run the simulation

def simulations(NUM_AGENTS):
    round_num = 1
    city_center = (0, 0)

    # Generate drivers only in round 1
    drivers1 = generate_drivers(NUM_AGENTS)
    drivers2 = copy.deepcopy(drivers1)
    drivers3 = copy.deepcopy(drivers1)
    drivers4 = copy.deepcopy(drivers1)


    # Generate passengers for round 1
    passengers = generate_passengers(NUM_AGENTS)
    passengers2 = copy.deepcopy(passengers)

    matches1 = driver_proposing_matching(drivers1, passengers, city_center, 0)
    matches2 = random_matches(drivers2, passengers)
    matches3 = closest_matches(drivers3, passengers)
    matches4 = driver_proposing_matching_new(drivers4, passengers2, city_center, 5)
    

    driver_dict = {driver.name: driver for driver in drivers1}
    passenger_dict = {passenger.name: passenger for passenger in passengers}
    passenger_dict_2 = {passenger.name: passenger for passenger in passengers2}


    for match in matches1:
        for driver in drivers1:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP

    for match in matches2:
        for driver in drivers2:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP
    
    for match in matches3:
        for driver in drivers3:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP

    for match in matches4:
        for driver in drivers4:
            if driver.name == match[0]:
                driver.matched_passenger_name = match[1]
            if driver.matched_passenger_name is not None:
                passenger = passenger_dict_2[driver.matched_passenger_name]
                driver.current_location = passenger.dropoff_location
                driver.total_income += passenger.WTP  

    total_len_1 = len(matches1)
    total_len_2 = len(matches2)  
    total_len_3 = len(matches3)  
    total_len_4 = len(matches4)            

    for round_num in range(2, 50):
        # Generate new passengers for each round
        # passengers = generate_passengers(random.randint(15,15))
        
        # Simulate the round
        new_passengers = generate_passengers(NUM_AGENTS)
        new_passengers2 = copy.deepcopy(new_passengers)
        matches1, length1 = simulate_round(drivers1, new_passengers, round_num, city_center, driver_proposing_matching, 0)
        matches2, length2 = simulate_round(drivers2, new_passengers, round_num, city_center, random_matches, 0)
        matches3, length3 = simulate_round(drivers3, new_passengers, round_num, city_center, closest_matches, 0)
        matches4, length4 = simulate_round(drivers4, new_passengers2, round_num, city_center, driver_proposing_matching_new, 5)

        total_len_1 += length1
        total_len_2 += length2
        total_len_3 += length3
        total_len_4 += length4

    # Print total_income values for each driver at the end of the simulation
    # print("\nIncome Stats:")
    drivers_list = [drivers1, drivers2, drivers3, drivers4]
    length_list = [total_len_1, total_len_2, total_len_3, total_len_4]
    results = {}
    for i in range(4):
        # for driver in drivers_list[i]:
            # print(f"{driver.name}: {driver.total_income}")
        total_incomes = [driver.total_income for driver in drivers_list[i]]
        revenue = sum(total_incomes)
        revenue_per_ride = revenue/length_list[i]
        mean_income = statistics.mean(total_incomes)
        sd_income = statistics.stdev(total_incomes)
        if i == 0:
            results["da"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        elif i == 1:
            results["random"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        elif i == 2:
            results["close"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        else:
            results["boston"] = [revenue, revenue_per_ride, mean_income, sd_income] 
        # print("\nRevenue of all drivers:" f"{i+1} "f"{revenue}")
        # print("\nMean income of all drivers:" f"{i+1} "f"{mean_income}")
        # print("\nSD income of all drivers:" f"{i+1} "f"{sd_income}")

    return results

    # print("\nSimulation completed!")


def main():
    incomes_da = []
    # incomes_ran = []
    # incomes_close = []
    incomes_hypothetically_fair = []

    for i in range(5,30):
        results = simulations(i)
        incomes_da.append(results["da"][3])
        # incomes_ran.append(results["random"][3])
        # incomes_close.append(results["close"][3])
        incomes_hypothetically_fair.append(results["boston"][3])
        

    # Sample data for multiple lines
    x_values = list(range(5, 30))
    y_values1 = incomes_da
    y_values2 = incomes_hypothetically_fair
    # y_values3 = incomes_close
    # y_values4 = incomes_boston

    # Plotting multiple lines
    plt.plot(x_values, y_values1, label='DA where drivers care more about proximity to passenger')
    plt.plot(x_values, y_values2, label='DA where drivers care more about dropoff location distance to center')
    # plt.plot(x_values, y_values3, label='Closeness')
    # plt.plot(x_values, y_values4, label='Boston')

    # Adding labels and title
    plt.title('SD for different weights')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()

    

# Run the main function if the script is executed
if __name__ == "__main__":
    main()


 
## Notes
# cost distance between driver current location, and passenger pickup location + some function of the euclidean distance between the passenger dropoff location and city center
# in practice drivers don't like far drop-off location (perhaps the probability of a passenger generating)
# show that we have considered these factors and for simplicity here is our function
## Suggestion: cost coefficient (not each driver has same cost) (different for each driver)
# also for revenue distribution, take into consideration how many rides they have picked up
# note in the writeup limitations (such as dynamic, with new drivers and passengers being generated)
# 2-3 papers
# the paper online bi-partite matching, each time a driver or rider comes in, each consumer has willingness to pay
