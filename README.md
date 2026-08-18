# Agent-Based Energy Behavior Model

An agent-based simulation of rumor-driven household energy behavior, electricity demand response, and solar PV adoption.

## Overview

This project investigates how social information, household characteristics, and policy incentives can influence residential energy behavior.

Households are represented as autonomous agents connected through a small-world social network. A rumor related to energy conservation or the benefits of solar PV spreads between neighboring households and can affect both electricity consumption and technology adoption.

The model demonstrates how individual-level behavioral decisions can generate system-level changes in energy demand.

## Model Structure

Each household is characterized by:

* baseline electricity consumption
* income level
* innovativeness
* solar PV capacity
* initial PV ownership
* rumor state

Households can occupy one of three rumor states:

* **S — Susceptible:** unaware of the information
* **I — Informed:** aware but not yet convinced
* **R — Believer:** accepts the information and changes behavior

### Behavioral Rules

At each simulation step:

1. Believing households can transmit the rumor to neighboring susceptible households.
2. Informed households can become believers with a specified probability.
3. Believers reduce their electricity consumption according to their income group.
4. Believers may adopt solar PV depending on:

   * individual innovativeness
   * social influence from neighboring adopters
   * income level
   * policy subsidy
5. PV generation is deducted from household electricity demand to calculate net load.

## Social Network

Households are connected through a **Watts–Strogatz small-world network**.

This network structure allows information and technology adoption to propagate both locally and through occasional long-range social connections.

## Technologies

* Python
* Mesa 2.1.1
* NetworkX
* NumPy
* pandas
* Matplotlib

## Repository Structure

```text
agent-based-energy-behavior-model/
│
├── AgentBasedModeling.py
├── households.csv
├── requirements.txt
└── README.md
```

## Input Data

The model reads household characteristics from `households.csv`.

The dataset contains the following variables:

* household ID
* baseline electricity demand
* income level
* innovativeness
* daily PV generation capacity
* initial PV ownership
* initial rumor state

The current dataset contains 10 synthetic households and is intended for demonstration and educational purposes.

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Running the Model

Example:

```bash
python AgentBasedModeling.py --csv households.csv --steps 60 --subsidy 0.1
```

The simulation parameters can also be modified from the command line, including:

* number of simulation steps
* policy subsidy
* rumor transmission probability
* belief probability
* social influence weight
* network connectivity
* network rewiring probability
* random seed

## Outputs

The model tracks:

* number of susceptible households
* number of informed households
* number of believers
* number of solar PV adopters
* total electricity demand
* average household electricity demand

Results are exported to a CSV file and visualized using Matplotlib.

Two primary plots are generated:

1. Evolution of believers, PV adopters, and total electricity demand
2. Dynamics of the S/I/R rumor states

## Example Simulation Results

In the example simulation with 10 households and a 10% subsidy:

* the rumor eventually spreads throughout the entire population;
* most households adopt solar PV;
* total net electricity demand decreases from approximately 125 kWh/day to approximately 65 kWh/day;
* the system reaches a relatively stable behavioral state after approximately 20 simulation steps.

These results illustrate the combined effects of behavioral change, social influence, and distributed solar adoption.

## Limitations

This model is primarily a conceptual and educational simulation.

Important limitations include:

* a small population of only 10 households;
* synthetic rather than empirical household data;
* simplified behavioral rules;
* fixed behavioral probabilities;
* limited representation of economic factors such as electricity prices and technology investment costs.

Therefore, the results should not be interpreted as predictions of real household behavior.

## Future Development

Possible extensions include:

* calibration using real household consumption data;
* larger and more realistic social networks;
* dynamic electricity prices;
* explicit PV investment costs;
* alternative subsidy policies;
* sensitivity and uncertainty analysis;
* heterogeneous behavioral parameters;
* comparison between different network structures.

## Purpose

This project was developed as an educational energy-system modeling exercise demonstrating the use of **agent-based modeling for analyzing behavioral and policy interactions in residential energy systems**.
