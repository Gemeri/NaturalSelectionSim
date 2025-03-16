"""
Each capsule lives on a limited flat plane. Red cylinders (food) are spawned periodically.
Capsules move about consuming energy (which increases with speed and body size according to a non-linear Kleiber‐like model).
They must eat food to survive; if their energy drops to 0 they die (starvation), if they exceed their maximum age they die (old_age),
or if a much larger capsule (≥25% larger) encounters them, it "eats" them (eaten). When a capsule accumulates enough energy,
it seeks a mate; when two similar‐sized capsules collide they reproduce. Offspring inherit the average gene values of the parents 
(with a small chance for mutation) using enhanced Mendelian inheritance (each gene is a genotype with two alleles; the dominant allele is expressed).
Additionally, reproduction may yield twins based on gene_twin_chance.

Capsule “color” was previously determined by gene contributions; this version omits any visual representation.

New graphs (saved in folder "info") every 10 seconds include:
  – Capsule Destinies (stacked histogram of Alive, Starvation, Old Age, Eaten),
  – Relative Death Rate by Age,
  – Time Series of the average Gene Speed, Gene Size, Reproduction Threshold Age, and Death Age,
  – A graph showing food availability and capsule population over time.
A CSV file ("capsule_data.csv") is updated with each capsule’s state and gene (genotype) values – now including an "Environment" attribute.
Autosave functionality saves the simulation every 10 minutes.

There are two distinct environments divided by a vertical wall at x=0.
  • Capsules with x < 0 are in environment "left" and receive abundant food.
  • Capsules with x ≥ 0 are in environment "right" and receive scarce food.
The wall is permanent so that capsules and food remain separated.
  
All simulation graphs and CSV outputs remain identical to before.
This version runs without any visual display for maximum speed.
"""

import random, math, os, csv, json, glob, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# ------------------ Minimal Vector Functions ------------------
# Instead of VPython’s vector and mag, we define simple versions using numpy.
def vector(x, y, z):
    return np.array([x, y, z], dtype=float)

def mag(v):
    return np.linalg.norm(v)

# ------------------ Base Gene Values and Helper Functions ------------------
BASE_SPEED = 8
BASE_CONSUMPTION = 0.5
BASE_SIZE = 1
BASE_SENSE = 15.0
BASE_MAX_AGE = 100
BASE_TWIN_CHANCE = 0.1

def create_genotype(base):
    return (base * random.uniform(0.9, 1.1), base * random.uniform(0.9, 1.1))

def express_genotype(genotype):
    return max(genotype)

def inherit_genotype(genotype1, genotype2, mutation_rate=0.05, base=1):
    allele1 = random.choice(genotype1)
    allele2 = random.choice(genotype2)
    if random.random() < mutation_rate:
        allele1 *= random.uniform(0.9, 1.1)
    if random.random() < mutation_rate:
        allele2 *= random.uniform(0.9, 1.1)
    return (allele1, allele2)

# ------------------ Simulation Parameters ------------------
dt = 0.1                         # Time step (seconds)
simulation_area = 50             # Half-width of simulation area

# Remove any VPython scene settings
# (Everything is now non-visual and runs on the command line)

# Food and graph parameters (randomized for each run)
food_energy_gain = random.randint(10, 60)  # Not used globally; re-randomized per event
food_spawn_interval = random.uniform(0.1, 10)   # randomized per spawn event
food_spawn_count = random.randint(1, 50)          # randomized per spawn event
graph_update_interval = 10.0                      # Update graphs every 10 seconds

# Capsule numbers per environment:
left_capsule_count = 5
right_capsule_count = 5
starting_capsule_number = left_capsule_count + right_capsule_count

action_wait_time = 1.0

# Energy parameters:  
# Each capsule gets random starting energy between 50 and 100 (set per capsule).
reproduction_threshold = 100
reproduction_cost = 50
reproduction_age_threshold = 16

food_collision_distance = 1.5
reproduction_collision_distance = 2.0

global_time = 0.0
last_food_spawn_time = 0.0
last_graph_update_time = 0.0
prediction_interval = 300 
last_prediction_time = 0.0

# Separate food spawn trackers for left/right (before wall removal)
last_food_spawn_time_left = 0.0
last_food_spawn_time_right = 0.0

# ------------------ Permanent Wall Setup ------------------
# In the non-visual version we do not create a visual wall but we retain its logic.
# Capsules with x < 0 are in "left" and x ≥ 0 in "right". The wall never disappears.
wall_position = 0  # x = 0 is the wall

# ------------------ Data Collections ------------------
dead_capsules_data = []
reproduction_ages = []
twin_reproduction_events = []

capsules = []
foods = []

avg_time_history = []
avg_speed_history = []
avg_consumption_history = []
avg_sense_history = []
avg_size_history = []
avg_twin_history = []
avg_repro_thresh_history = []
avg_death_age_history = []
population_history = []
food_count_history = []

left_avg_size_history = []
right_avg_size_history = []
left_avg_speed_history = []
right_avg_speed_history = []
left_avg_consumption_history = []
right_avg_consumption_history = []

# ------------------ Time Series Update Functions ------------------
def update_time_series():
    alive_caps = [cap for cap in capsules if cap.alive]
    population_history.append(len(alive_caps))
    food_count_history.append(sum(1 for f in foods if f.active))
    if not alive_caps:
        return
    avg_speed = np.mean([cap.gene_speed for cap in alive_caps])
    avg_consumption = np.mean([cap.gene_consumption for cap in alive_caps])
    avg_sense = np.mean([cap.gene_sense for cap in alive_caps])
    avg_size = np.mean([cap.gene_size for cap in alive_caps])
    avg_twin = np.mean([cap.gene_twin_chance for cap in alive_caps])
    repro_thresholds = [reproduction_age_threshold * (cap.gene_max_age/BASE_MAX_AGE) for cap in alive_caps]
    avg_repro_thresh = np.mean(repro_thresholds)
    
    avg_time_history.append(global_time)
    avg_speed_history.append(avg_speed)
    avg_consumption_history.append(avg_consumption)
    avg_sense_history.append(avg_sense)
    avg_size_history.append(avg_size)
    avg_twin_history.append(avg_twin)
    avg_repro_thresh_history.append(avg_repro_thresh)
    
    avg_death_age = np.mean([data["age"] for data in dead_capsules_data]) if dead_capsules_data else 0
    avg_death_age_history.append(avg_death_age)
    
    if not os.path.exists("info"):
        os.makedirs("info")
    
    pd.DataFrame({"Time": avg_time_history, "Average_Gene_Speed": avg_speed_history}).to_csv("info/avg_speed_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Gene_Consumption": avg_consumption_history}).to_csv("info/avg_consumption_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Gene_Sense": avg_sense_history}).to_csv("info/avg_sense_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Gene_Size": avg_size_history}).to_csv("info/avg_size_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Gene_Twin_Chance": avg_twin_history}).to_csv("info/avg_twin_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Reproduction_Threshold_Age": avg_repro_thresh_history}).to_csv("info/avg_repro_thresh_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Average_Death_Age": avg_death_age_history}).to_csv("info/avg_death_age_timeseries.csv", index=False)
    pd.DataFrame({"Time": avg_time_history, "Capsule_Population": population_history, "Food_Count": food_count_history}).to_csv("info/population_food_timeseries.csv", index=False)

def update_env_time_series():
    left_caps = [cap for cap in capsules if cap.alive and cap.environment == "left"]
    right_caps = [cap for cap in capsules if cap.alive and cap.environment == "right"]
    
    if left_caps:
        left_avg_size = np.mean([cap.gene_size for cap in left_caps])
        left_avg_speed = np.mean([cap.gene_speed for cap in left_caps])
        left_avg_consumption = np.mean([cap.gene_consumption for cap in left_caps])
    else:
        left_avg_size = left_avg_speed = left_avg_consumption = 0
        
    if right_caps:
        right_avg_size = np.mean([cap.gene_size for cap in right_caps])
        right_avg_speed = np.mean([cap.gene_speed for cap in right_caps])
        right_avg_consumption = np.mean([cap.gene_consumption for cap in right_caps])
    else:
        right_avg_size = right_avg_speed = right_avg_consumption = 0

    left_avg_size_history.append(left_avg_size)
    right_avg_size_history.append(right_avg_size)
    left_avg_speed_history.append(left_avg_speed)
    right_avg_speed_history.append(right_avg_speed)
    left_avg_consumption_history.append(left_avg_consumption)
    right_avg_consumption_history.append(right_avg_consumption)

    plt.figure()
    plt.plot(avg_time_history, left_avg_size_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Size (Left)")
    plt.title("Left Avg Size Timeseries")
    plt.savefig("info/left_avg_size_timeseries.png")
    plt.close()

    plt.figure()
    plt.plot(avg_time_history, right_avg_size_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Size (Right)")
    plt.title("Right Avg Size Timeseries")
    plt.savefig("info/right_avg_size_timeseries.png")
    plt.close()

    plt.figure()
    plt.plot(avg_time_history, left_avg_speed_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Speed (Left)")
    plt.title("Left Avg Speed Timeseries")
    plt.savefig("info/left_avg_speed_timeseries.png")
    plt.close()

    plt.figure()
    plt.plot(avg_time_history, right_avg_speed_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Speed (Right)")
    plt.title("Right Avg Speed Timeseries")
    plt.savefig("info/right_avg_speed_timeseries.png")
    plt.close()

    plt.figure()
    plt.plot(avg_time_history, left_avg_consumption_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Consumption (Left)")
    plt.title("Left Avg Consumption Timeseries")
    plt.savefig("info/left_avg_consumption_timeseries.png")
    plt.close()

    plt.figure()
    plt.plot(avg_time_history, right_avg_consumption_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Avg Gene Consumption (Right)")
    plt.title("Right Avg Consumption Timeseries")
    plt.savefig("info/right_avg_consumption_timeseries.png")
    plt.close()

# ------------------ Continuous Bar Graphs for Gene Speed (Environment Specific) ------------------
def update_continuous_graphs(current_time):
    alive_caps = [cap for cap in capsules if cap.alive]
    if not alive_caps:
        return
    if not os.path.exists("continuous"):
        os.makedirs("continuous")
    for env, tag in [("left", "left"), ("right", "right")]:
        env_caps = [cap for cap in alive_caps if cap.environment == env]
        if not env_caps:
            continue
        values = [cap.gene_speed for cap in env_caps]
        plt.figure()
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            min_val -= 0.5; max_val += 0.5
        bins = np.linspace(min_val, max_val, 15)
        plt.hist(values, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel("Gene Speed")
        plt.ylabel("Count")
        plt.title(f"Distribution of Gene Speed ({tag.capitalize()})")
        plt.savefig(f"continuous/{tag}_gene_speed_{int(current_time)}.png")
        plt.close()

# ------------------ Display Graphs ------------------
# In the non-visual version we no longer display HTML or use a browser.
# Instead, we simply update and save the graphs.
def display_graphs():
    # This function used to generate an HTML window.
    # Now, it does nothing extra since the graphs are saved to disk.
    pass

# ------------------ Capsule Class with Enhanced Genetic Modeling ------------------
class Capsule:
    def __init__(self, pos, genes=None):
        self.pos = vector(pos[0], pos[1], pos[2]) if isinstance(pos, (list, tuple, np.ndarray)) else pos.copy()
        self.environment = "left" if self.pos[0] < wall_position else "right"
        if genes is None:
            self.genotype_speed = create_genotype(BASE_SPEED)
            self.genotype_size = create_genotype(BASE_SIZE)
            self.genotype_sense = create_genotype(BASE_SENSE)
            self.genotype_max_age = create_genotype(BASE_MAX_AGE)
            self.genotype_twin_chance = create_genotype(BASE_TWIN_CHANCE)
        else:
            self.genotype_speed = tuple(genes["genotype_speed"])
            self.genotype_size = tuple(genes["genotype_size"])
            self.genotype_sense = tuple(genes["genotype_sense"])
            self.genotype_max_age = tuple(genes["genotype_max_age"])
            self.genotype_twin_chance = tuple(genes["genotype_twin_chance"])
        self.gene_speed = express_genotype(self.genotype_speed)
        self.gene_size = express_genotype(self.genotype_size)
        self.gene_sense = express_genotype(self.genotype_sense)
        self.gene_max_age = express_genotype(self.genotype_max_age)
        self.gene_twin_chance = express_genotype(self.genotype_twin_chance)
        # Compute consumption using non-linear Kleiber-like scaling (from size and speed only)
        self.gene_consumption = 0.2 * (self.gene_size ** 2.25) * self.gene_speed

        self.energy = random.randint(50, 100)
        self.age = 0.0
        self.wait_timer = 0.0
        self.alive = True
        self.death_cause = None

        angle = random.uniform(0, 2*math.pi)
        self.velocity = self.gene_speed * vector(math.cos(angle), 0, math.sin(angle))

        # Visual elements removed for speed; no drawing is performed.
        # Instead, all computations use numerical vectors.
        # (update_color is now a no-op.)
        self.update_color()

    def update_color(self):
        # No visual element; method retained for compatibility.
        pass

    def update(self, dt):
        if not self.alive:
            return
        self.age += dt
        self.energy -= self.gene_consumption * dt
        if self.age > self.gene_max_age:
            self.die("old_age")
            return
        if self.wait_timer > 0:
            self.wait_timer -= dt
            return

        # Since the wall is permanent, we always use environment-based interactions.
        def same_environment(entity):
            return hasattr(entity, "environment") and (entity.environment == self.environment)

        threat = None
        threat_distance = None
        for other in capsules:
            if other is not self and other.alive and same_environment(other):
                if other.gene_size >= 1.25 * self.gene_size:
                    d = mag(other.pos - self.pos)
                    if d <= self.gene_sense and (threat is None or d < threat_distance):
                        threat = other
                        threat_distance = d
        if threat is not None:
            target_direction = self.pos - threat.pos
            if mag(target_direction) != 0:
                self.velocity = self.gene_speed * (target_direction / mag(target_direction))
        else:
            target_direction = vector(0, 0, 0)
            target_found = False
            if self.energy < reproduction_threshold:
                nearest_food = None
                nearest_food_distance = None
                for f in foods:
                    if f.active and same_environment(f):
                        d = mag(f.pos - self.pos)
                        if d <= self.gene_sense and (nearest_food is None or d < nearest_food_distance):
                            nearest_food = f
                            nearest_food_distance = d
                if nearest_food is not None:
                    target_direction = nearest_food.pos - self.pos
                    target_found = True
            else:
                nearest_partner = None
                nearest_partner_distance = None
                for other in capsules:
                    if other is not self and other.alive and other.energy >= reproduction_threshold and other.wait_timer <= 0 and same_environment(other):
                        d = mag(other.pos - self.pos)
                        if d <= self.gene_sense and (nearest_partner is None or d < nearest_partner_distance):
                            nearest_partner = other
                            nearest_partner_distance = d
                if nearest_partner is not None:
                    target_direction = nearest_partner.pos - self.pos
                    target_found = True
            if target_found and mag(target_direction) != 0:
                self.velocity = self.gene_speed * (target_direction / mag(target_direction))
            else:
                angle_change = random.uniform(-0.2, 0.2)
                speed = mag(self.velocity)
                current_angle = math.atan2(self.velocity[2], self.velocity[0])
                new_angle = current_angle + angle_change
                self.velocity = speed * vector(math.cos(new_angle), 0, math.sin(new_angle))
        self.pos += self.velocity * dt

        # Enforce wall boundaries (capsules cannot cross x=0)
        if self.environment == "left" and self.pos[0] > wall_position:
            self.pos[0] = wall_position
            self.velocity[0] = 0
        elif self.environment == "right" and self.pos[0] < wall_position:
            self.pos[0] = wall_position
            self.velocity[0] = 0

        # Enforce simulation area boundaries
        if abs(self.pos[0]) > simulation_area:
            self.velocity[0] = -self.velocity[0]
            self.pos[0] = math.copysign(simulation_area, self.pos[0])
        if abs(self.pos[2]) > simulation_area:
            self.velocity[2] = -self.velocity[2]
            self.pos[2] = math.copysign(simulation_area, self.pos[2])

    def eat(self, food):
        gain = random.randint(10, 80)
        self.energy += gain
        food.active = False
        self.wait_timer = action_wait_time

    def can_reproduce(self):
        return self.energy >= reproduction_threshold

    def reproduce(self):
        self.energy -= reproduction_cost
        self.wait_timer = action_wait_time

    def die(self, cause):
        self.alive = False
        self.death_cause = cause
        dead_capsules_data.append({
            "age": self.age,
            "energy": self.energy,
            "speed": self.gene_speed,
            "consumption": self.gene_consumption,
            "size": self.gene_size,
            "sense": self.gene_sense,
            "max_age": self.gene_max_age,
            "twin": self.gene_twin_chance,
            "cause": cause,
            "environment": self.environment
        })

# ------------------ Food Class ------------------
class Food:
    def __init__(self, pos):
        self.pos = vector(pos[0], pos[1], pos[2]) if isinstance(pos, (list, tuple, np.ndarray)) else pos.copy()
        # Instead of a visual, we use an "active" flag.
        self.active = True
        # Environment assignment remains (based on x position) so that food spawns appropriately.
        self.environment = "left" if self.pos[0] < wall_position else "right"

# ------------------ Spawning Functions ------------------
def spawn_food():
    global last_food_spawn_time_left, last_food_spawn_time_right, last_food_spawn_time
    # Always use the left/right environment logic (wall never disappears)
    # Left side: abundant food.
    left_interval = random.uniform(0.5, 2)
    if global_time - last_food_spawn_time_left > left_interval:
        left_count = random.randint(10, 20)
        for i in range(left_count):
            x = random.uniform(-simulation_area, wall_position)
            z = random.uniform(-simulation_area, simulation_area)
            f = Food(vector(x, 0, z))
            f.environment = "left"
            foods.append(f)
        last_food_spawn_time_left = global_time
    # Right side: scarce food.
    right_interval = random.uniform(3, 10)
    if global_time - last_food_spawn_time_right > right_interval:
        right_count = random.randint(1, 20)
        for i in range(right_count):
            x = random.uniform(wall_position, simulation_area)
            z = random.uniform(-simulation_area, simulation_area)
            f = Food(vector(x, 0, z))
            f.environment = "right"
            foods.append(f)
        last_food_spawn_time_right = global_time

def spawn_capsule(pos, genes=None):
    cap = Capsule(pos, genes)
    capsules.append(cap)

def reproduce_parents(parent1, parent2, pos):
    mutation_rate = 0.05
    child_genotype_speed = inherit_genotype(parent1.genotype_speed, parent2.genotype_speed, mutation_rate, BASE_SPEED)
    child_genotype_size = inherit_genotype(parent1.genotype_size, parent2.genotype_size, mutation_rate, BASE_SIZE)
    child_genotype_sense = inherit_genotype(parent1.genotype_sense, parent2.genotype_sense, mutation_rate, BASE_SENSE)
    child_genotype_max_age = inherit_genotype(parent1.genotype_max_age, parent2.genotype_max_age, mutation_rate, BASE_MAX_AGE)
    child_genotype_twin = inherit_genotype(parent1.genotype_twin_chance, parent2.genotype_twin_chance, mutation_rate, BASE_TWIN_CHANCE)
    child_genes = {
        "genotype_speed": child_genotype_speed,
        "genotype_size": child_genotype_size,
        "genotype_sense": child_genotype_sense,
        "genotype_max_age": child_genotype_max_age,
        "genotype_twin_chance": child_genotype_twin
    }
    return Capsule(pos, genes=child_genes)

# ------------------ Initial Spawning ------------------
for i in range(left_capsule_count):
    x = random.uniform(-simulation_area, wall_position)
    z = random.uniform(-simulation_area, simulation_area)
    spawn_capsule(vector(x, 0, z))
for i in range(right_capsule_count):
    x = random.uniform(wall_position, simulation_area)
    z = random.uniform(-simulation_area, simulation_area)
    spawn_capsule(vector(x, 0, z))

# ------------------ CSV Update Functions ------------------
def update_alive_avg_csv():
    alive_caps = [cap for cap in capsules if cap.alive]
    if not alive_caps:
        return
    avg_speed = np.mean([cap.gene_speed for cap in alive_caps])
    avg_consumption = np.mean([cap.gene_consumption for cap in alive_caps])
    avg_size = np.mean([cap.gene_size for cap in alive_caps])
    avg_sense = np.mean([cap.gene_sense for cap in alive_caps])
    avg_max_age = np.mean([cap.gene_max_age for cap in alive_caps])
    avg_twin = np.mean([cap.gene_twin_chance for cap in alive_caps])
    avg_repro_thresh = np.mean([reproduction_age_threshold * (cap.gene_max_age/BASE_MAX_AGE) for cap in alive_caps])
    with open("alive_avg.csv", "w", newline="") as csvfile:
        fieldnames = ["Average_Gene_Speed", "Average_Gene_Consumption", "Average_Gene_Size",
                      "Average_Gene_Sense", "Average_Gene_Max_Age", "Average_Gene_Twin_Chance",
                      "Average_Reproduction_Threshold_Age"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Average_Gene_Speed": avg_speed,
            "Average_Gene_Consumption": avg_consumption,
            "Average_Gene_Size": avg_size,
            "Average_Gene_Sense": avg_sense,
            "Average_Gene_Max_Age": avg_max_age,
            "Average_Gene_Twin_Chance": avg_twin,
            "Average_Reproduction_Threshold_Age": avg_repro_thresh
        })

def predict_future_genes():
    try:
        df = pd.read_csv("capsule_data.csv")
    except Exception:
        return
    predictions = {}
    genes = ["Gene_Speed", "Gene_Consumption", "Gene_Size", "Gene_Sense", "Gene_Max_Age", "Gene_Twin_Chance"]
    if len(df) == 0:
        return
    future_age = df["Age"].max() + 10
    for gene in genes:
        if gene not in df.columns:
            continue
        X = df[["Age"]]
        y = df[gene]
        if len(X) < 2:
            continue
        # Use GPU acceleration (Tesla T4) with xgboost if available
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, tree_method='gpu_hist', gpu_id=0)
        model.fit(X, y)
        pred = model.predict([[future_age]])[0]
        predictions[gene] = pred
    if predictions:
        with open("predicted_genes.csv", "w", newline="") as csvfile:
            fieldnames = list(predictions.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(predictions)

def plot_time_series():
    plt.figure()
    plt.plot(avg_time_history, avg_repro_thresh_history, marker='o', label="Reproduction Threshold Age")
    plt.plot(avg_time_history, avg_death_age_history, marker='o', label="Death Age")
    plt.xlabel("Time (s)")
    plt.ylabel("Age")
    plt.title("Reproduction Threshold and Death Age Over Time")
    plt.legend()
    plt.savefig("info/avg_age_timeseries.png")
    plt.close()
    
    plt.figure()
    plt.plot(avg_time_history, avg_speed_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Average Gene Speed")
    plt.title("Average Gene Speed Over Time")
    plt.savefig("info/avg_speed_timeseries.png")
    plt.close()
    
    plt.figure()
    plt.plot(avg_time_history, avg_size_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Average Gene Size")
    plt.title("Average Gene Size Over Time")
    plt.savefig("info/avg_size_timeseries.png")
    plt.close()

def update_graphs(current_time):
    alive_ages = [cap.age for cap in capsules if cap.alive]
    dead_ages = [data["age"] for data in dead_capsules_data]
    all_ages = dead_ages + alive_ages
    if not all_ages:
        return
    bin_width = 5
    max_age_val = max(all_ages) if all_ages else 50
    bins = np.arange(0, max_age_val + bin_width, bin_width)
    
    death_starvation = [data["age"] for data in dead_capsules_data if data["cause"]=="starvation"]
    death_old_age = [data["age"] for data in dead_capsules_data if data["cause"]=="old_age"]
    death_eaten = [data["age"] for data in dead_capsules_data if data["cause"]=="eaten"]

    if not os.path.exists("info"):
        os.makedirs("info")
    plt.figure()
    plt.hist([alive_ages, death_starvation, death_old_age, death_eaten],
             bins=bins, stacked=True,
             label=["Alive", "Starvation", "Old Age", "Eaten"])
    plt.xlabel("Age")
    plt.ylabel("Population Count")
    plt.title("Capsule Destinies (Alive vs Dead by Cause)")
    plt.legend()
    plt.savefig("info/capsule_destinies.png")
    plt.close()

    # Compute relative death rate by age.
    at_risk = []
    death_counts = []
    for i in range(len(bins)-1):
        age_lower = bins[i]
        n_at_risk = sum(1 for cap in capsules if cap.alive and cap.age >= age_lower) + \
                    sum(1 for data in dead_capsules_data if data["age"] >= age_lower)
        n_deaths = sum(1 for data in dead_capsules_data if age_lower <= data["age"] < bins[i+1])
        at_risk.append(n_at_risk)
        death_counts.append(n_deaths)
    hazard = [n_d / n_r if n_r > 0 else 0 for n_d, n_r in zip(death_counts, at_risk)]
    plt.figure()
    plt.bar(bins[:-1], hazard, width=bin_width, align='edge')
    plt.ylim(0, 1)
    plt.xlabel("Age")
    plt.ylabel("Relative Death Rate")
    plt.title("Relative Death Rate by Age")
    plt.savefig("info/relative_death_rate.png")
    plt.close()

    update_time_series()
    plot_time_series()
    update_env_time_series()
    update_continuous_graphs(current_time)
    display_graphs()

def update_csv():
    with open("capsule_data.csv", "w", newline="") as csvfile:
        fieldnames = ["CapsuleNumber", "Energy", "Age", "Gene_Speed", "Gene_Consumption",
                      "Gene_Size", "Gene_Sense", "Gene_Max_Age", "Gene_Twin_Chance", "Death_Cause", "Environment"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        n = 1
        for cap in capsules:
            writer.writerow({
                "CapsuleNumber": n,
                "Energy": cap.energy,
                "Age": cap.age,
                "Gene_Speed": cap.gene_speed,
                "Gene_Consumption": cap.gene_consumption,
                "Gene_Size": cap.gene_size,
                "Gene_Sense": cap.gene_sense,
                "Gene_Max_Age": cap.gene_max_age,
                "Gene_Twin_Chance": cap.gene_twin_chance,
                "Death_Cause": "" if cap.alive else cap.death_cause,
                "Environment": cap.environment
            })
            n += 1
        for data in dead_capsules_data:
            writer.writerow({
                "CapsuleNumber": n,
                "Energy": data["energy"],
                "Age": data["age"],
                "Gene_Speed": data["speed"],
                "Gene_Consumption": data["consumption"],
                "Gene_Size": data["size"],
                "Gene_Sense": data["sense"],
                "Gene_Max_Age": data["max_age"],
                "Gene_Twin_Chance": data["twin"],
                "Death_Cause": data["cause"],
                "Environment": data.get("environment", "")
            })
            n += 1

# ------------------ Save/Load Functions ------------------
def save_simulation():
    state = {
         "global_time": global_time,
         "last_food_spawn_time": last_food_spawn_time,
         "last_graph_update_time": last_graph_update_time,
         "last_prediction_time": last_prediction_time,
         "parameters": {
              "food_spawn_interval": food_spawn_interval,
              "food_spawn_count": food_spawn_count,
              "initial_energy": "random per capsule",
              "food_energy_gain": food_energy_gain,
              "reproduction_threshold": reproduction_threshold,
              "reproduction_cost": reproduction_cost,
              "reproduction_age_threshold": reproduction_age_threshold
         },
         "capsules": [],
         "foods": [],
         "dead_capsules_data": dead_capsules_data,
         "reproduction_ages": reproduction_ages,
         "twin_reproduction_events": twin_reproduction_events,
         "avg_time_history": avg_time_history,
         "avg_speed_history": avg_speed_history,
         "avg_consumption_history": avg_consumption_history,
         "avg_sense_history": avg_sense_history,
         "avg_size_history": avg_size_history,
         "avg_twin_history": avg_twin_history,
         "avg_repro_thresh_history": avg_repro_thresh_history,
         "avg_death_age_history": avg_death_age_history,
         "population_history": population_history,
         "food_count_history": food_count_history
    }
    for cap in capsules:
        cap_state = {
            "pos": cap.pos.tolist(),
            "energy": cap.energy,
            "age": cap.age,
            "wait_timer": cap.wait_timer,
            "alive": cap.alive,
            "genotype_speed": list(cap.genotype_speed),
            "genotype_size": list(cap.genotype_size),
            "genotype_sense": list(cap.genotype_sense),
            "genotype_max_age": list(cap.genotype_max_age),
            "genotype_twin_chance": list(cap.genotype_twin_chance)
        }
        state["capsules"].append(cap_state)
    for f in foods:
        food_state = {"pos": f.pos.tolist()}
        state["foods"].append(food_state)
    with open("simulation_save.json", "w") as fh:
        json.dump(state, fh)
    print("Simulation saved.")

def load_simulation():
    global global_time, last_food_spawn_time, last_graph_update_time, last_prediction_time
    global food_spawn_interval, food_spawn_count, food_energy_gain
    global reproduction_threshold, reproduction_cost, reproduction_age_threshold
    global capsules, foods, dead_capsules_data, reproduction_ages, twin_reproduction_events
    global avg_time_history, avg_speed_history, avg_consumption_history, avg_sense_history, avg_size_history, avg_twin_history, avg_repro_thresh_history, avg_death_age_history
    global population_history, food_count_history

    # Since visuals are not used, we simply clear our lists.
    capsules = []
    foods = []
    dead_capsules_data = []
    reproduction_ages = []
    twin_reproduction_events = []
    avg_time_history = []
    avg_speed_history = []
    avg_consumption_history = []
    avg_sense_history = []
    avg_size_history = []
    avg_twin_history = []
    avg_repro_thresh_history = []
    avg_death_age_history = []
    population_history = []
    food_count_history = []

    try:
        with open("simulation_save.json", "r") as fh:
            state = json.load(fh)
    except Exception as e:
        print("Failed to load simulation:", e)
        return

    global_time = state["global_time"]
    last_food_spawn_time = state["last_food_spawn_time"]
    last_graph_update_time = state["last_graph_update_time"]
    last_prediction_time = state["last_prediction_time"]
    params = state["parameters"]

    food_spawn_interval = params["food_spawn_interval"]
    food_spawn_count = params["food_spawn_count"]
    food_energy_gain = params["food_energy_gain"]
    reproduction_threshold = params["reproduction_threshold"]
    reproduction_cost = params["reproduction_cost"]
    reproduction_age_threshold = params["reproduction_age_threshold"]

    for cap_state in state["capsules"]:
        pos = vector(*cap_state["pos"])
        genes = {
            "genotype_speed": cap_state["genotype_speed"],
            "genotype_size": cap_state["genotype_size"],
            "genotype_sense": cap_state["genotype_sense"],
            "genotype_max_age": cap_state["genotype_max_age"],
            "genotype_twin_chance": cap_state["genotype_twin_chance"]
        }
        new_cap = Capsule(pos, genes=genes)
        new_cap.energy = cap_state["energy"]
        new_cap.age = cap_state["age"]
        new_cap.wait_timer = cap_state["wait_timer"]
        new_cap.alive = cap_state["alive"]
        capsules.append(new_cap)

    for f_state in state["foods"]:
        pos = vector(*f_state["pos"])
        foods.append(Food(pos))

    dead_capsules_data = state["dead_capsules_data"]
    reproduction_ages = state["reproduction_ages"]
    twin_reproduction_events = state["twin_reproduction_events"]
    avg_time_history = state.get("avg_time_history", [])
    avg_speed_history = state.get("avg_speed_history", [])
    avg_consumption_history = state.get("avg_consumption_history", [])
    avg_sense_history = state.get("avg_sense_history", [])
    avg_size_history = state.get("avg_size_history", [])
    avg_twin_history = state.get("avg_twin_history", [])
    avg_repro_thresh_history = state.get("avg_repro_thresh_history", [])
    avg_death_age_history = state.get("avg_death_age_history", [])
    population_history = state.get("population_history", [])
    food_count_history = state.get("food_count_history", [])

    print("Simulation loaded.")

# ------------------ Autosave Functionality ------------------
autosave_interval = 600
last_autosave_time = 0.0

def auto_save_simulation():
    global last_autosave_time, global_time
    if global_time - last_autosave_time >= autosave_interval:
        save_simulation()
        last_autosave_time = global_time
        print("Autosave completed at time:", global_time)

# ------------------ Main Simulation Loop ------------------
# (No visual update or pause buttons are used in this command-line version)
print("Starting simulation...")
while True:
    # Advance simulation time
    global_time += dt

    # Update each capsule (behavior, movement, energy, age, etc.)
    for cap in capsules:
        if cap.alive:
            cap.update(dt)

    # Process food consumption:
    for cap in capsules:
        if not cap.alive or cap.wait_timer > 0:
            continue
        for f in foods[:]:
            if f.active:
                # Ensure food and capsule are in the same environment.
                if f.environment != cap.environment:
                    continue
                if mag(cap.pos - f.pos) < food_collision_distance:
                    cap.eat(f)
                    foods.remove(f)

    # Process capsule collisions for reproduction and predation:
    for i in range(len(capsules)):
        for j in range(i+1, len(capsules)):
            cap1 = capsules[i]
            cap2 = capsules[j]
            # Capsules in different environments do not interact.
            if cap1.environment != cap2.environment:
                continue
            if cap1.alive and cap2.alive and cap1.wait_timer <= 0 and cap2.wait_timer <= 0:
                if mag(cap1.pos - cap2.pos) < reproduction_collision_distance:
                    if cap1.gene_size >= 1.25 * cap2.gene_size:
                        food_energy_gain = random.randint(20, 80)
                        cap1.energy += food_energy_gain
                        cap2.die("eaten")
                    elif cap2.gene_size >= 1.25 * cap1.gene_size:
                        food_energy_gain = random.randint(20, 80)
                        cap2.energy += food_energy_gain
                        cap1.die("eaten")
                    else:
                        if (cap1.age >= reproduction_age_threshold * (cap1.gene_max_age/BASE_MAX_AGE)
                            and cap2.age >= reproduction_age_threshold * (cap2.gene_max_age/BASE_MAX_AGE)):
                            avg_twin = (cap1.gene_twin_chance + cap2.gene_twin_chance) / 2
                            twins = (random.random() < avg_twin)
                            offspring = reproduce_parents(cap1, cap2, (cap1.pos + cap2.pos)/2)
                            capsules.append(offspring)
                            if twins:
                                twin_offspring = reproduce_parents(cap1, cap2, (cap1.pos + cap2.pos)/2)
                                capsules.append(twin_offspring)
                                twin_reproduction_events.append((cap1.age + cap2.age)/2)
                            cap1.reproduce()
                            cap2.reproduce()

    # Process starvation: if a capsule's energy has dropped to zero or below, it dies.
    for cap in capsules[:]:
        if cap.alive and cap.energy <= 0:
            cap.die("starvation")
            capsules.remove(cap)

    # Spawn new food (using permanent wall logic)
    spawn_food()

    # Update graphs and CSV files every graph_update_interval seconds
    if global_time - last_graph_update_time > graph_update_interval:
        update_graphs(global_time)
        update_csv()
        update_alive_avg_csv()
        last_graph_update_time = global_time

    # Update gene predictions every prediction_interval seconds
    if global_time - last_prediction_time > prediction_interval:
        predict_future_genes()
        last_prediction_time = global_time

    # Autosave simulation state every autosave_interval seconds
    auto_save_simulation()
    
    # (No artificial sleep here to maximize simulation speed.)
