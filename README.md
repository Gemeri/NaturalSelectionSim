# Natural Selection Simulation

This project is a 3D natural selection simulation built using VPython and Python. The simulation models capsule‐like organisms that live in a divided environment, evolve via enhanced Mendelian genetics, consume energy according to a non‐linear (Kleiber-like) metabolic model, and interact with food and one another. It includes two distinct environments with different food availabilities that merge after 4 hours of simulation time.

## Overview

- **Two Distinct Environments:**  
  The simulation space is split by a vertical, semi‑transparent wall at `x=0`. Before 4 hours (14,400 seconds) capsules and food remain confined to their own half:
  - **Left Environment:** Capsules with `x < 0` with abundant food (frequent spawns).
  - **Right Environment:** Capsules with `x ≥ 0` with scarce food (infrequent spawns).  
  Capsules and food are labeled by environment, and sensing & interactions are restricted to within the same half. After 4 hours, the wall is automatically removed, and all entities interact freely.

- **Enhanced Genetic Modeling:**  
  Each capsule has genes for speed, size, sense, maximum age, and twin reproduction chance. These are modeled as genotypes with two alleles (the dominant allele is expressed). Offspring inherit the average gene values (with mutation) from their parents.

- **Energy Consumption:**  
  Energy consumption follows a non‑linear Kleiber‑like model that scales with capsule body size and speed. Capsules must eat food to replenish energy and survive.

- **Dynamic Graphs and Data Collection:**  
  The simulation automatically updates several graphs every 10 seconds and writes CSV files containing real‑time data. New graphs include:
  - Capsule destinies (alive, starvation, old age, eaten)
  - Relative death rate by age
  - Time series of overall averages (gene speed, gene size, reproduction threshold age, death age)
  - Environment‑specific time series (left/right average gene size, gene speed, and gene consumption)
  - Continuous bar graphs for gene speed distributions for left and right environments
  - A graph showing food availability and capsule population over time

- **Interactive Display:**  
  Eight draggable and resizable graph windows appear next to the simulation. These display the environment‑specific time series and continuous bar graphs.

- **Save/Load & Autosave:**  
  The simulation supports pause, save, and load functionality and autosaves every 10 minutes.

- **Local HTTP Server Requirement:**  
  **Important:** To ensure that image URLs for graphs load correctly, run the simulation from a local HTTP server (e.g. via `python -m http.server 8000`).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Gemeri/NaturalSelectionSim
   cd NaturalSelectionSim
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.x installed. Then install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start a Local HTTP Server:**

   In your simulation folder, run:

   ```bash
   python -m http.server 8000
   ```

   This will serve your simulation files at `http://localhost:8000/`.

2. **Run the Simulation Script:**

   In a separate terminal (or the same if you use multitasking), run:

   ```bash
   python com.py
   ```

3. **Interacting with the Simulation:**

   - **Environments:**  
     Initially, capsules and food are separated by a vertical wall at `x=0` into left and right environments with different food spawn rates.
     
   - **Wall Removal:**  
     After 4 hours (14,400 seconds) of simulation time, the wall is automatically removed, and all capsules begin to interact freely.
     
   - **Data & Graphs:**  
     Graphs update every 10 seconds and are displayed in eight separate draggable/resizable windows on the side. The CSV file `capsule_data.csv` and various time series graphs are updated automatically.
     
   - **Save/Load & Autosave:**  
     Use the provided buttons to pause, save, or load the simulation. The simulation also autosaves every 10 minutes.

## Customization

- **Randomized Parameters:**  
  - Food spawn interval is randomized between 0.1 and 10 seconds.
  - Food spawn count is randomized between 1 and 50 items.
  - Food energy gain is randomized per event between 10 and 80.
  - Each capsule’s initial energy is randomly assigned between 50 and 100.

- **Environment-Specific Adjustments:**  
  You can modify the number of initial capsules per environment, and adjust the food spawn parameters for each side within the script.

- **Graph Windows:**  
  The eight separate graph windows display environment-specific averages (for gene speed, gene size, and gene consumption) as line graphs and continuous bar graphs for gene speed distribution. These windows are draggable and resizable.

## Code Structure

- **com.py:**  
  Contains the main simulation code including the VPython simulation, data collection, graph updates, save/load, and autosave functionality.

- **info/**  
  Contains CSV files and time series graphs (updated automatically).

- **continuous/**  
  Contains continuous bar graph images for gene speed distributions (updated automatically).
