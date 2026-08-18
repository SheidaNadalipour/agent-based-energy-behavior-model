# ABM of rumor-driven energy behavior (reads households.csv)
# Requirements: mesa==2.1.1, networkx, numpy, matplotlib, pandas
# Run:
#   python abm_from_csv.py
#   python abm_from_csv.py --csv households.csv --steps 60 --subsidy 0.1

from __future__ import annotations
import argparse
import random
from typing import Dict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------- Agent ----------------
class Household(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "RumorEnergyModel",
        income_level: str,
        base_load_kwh: float,
        innovativeness: float,
        pv_size_kwh_day: float,
        has_solar_init: bool,
        initial_state: str,
    ):
        super().__init__(unique_id, model)
        self.income_level = income_level.lower().strip()   # low|medium|high
        self.base_load = float(base_load_kwh)
        self.current_load = float(base_load_kwh)
        self.innovativeness = float(innovativeness)
        self.has_solar = bool(int(has_solar_init))
        self.pv_size_kwh_day = float(pv_size_kwh_day)
        self.rumor_state = initial_state.strip().upper()   # S|I|R

    def _maybe_progress_rumor(self):
        if self.rumor_state == "I" and random.random() < self.model.belief_prob:
            self.rumor_state = "R"

    def _spread_rumor(self):
        if self.rumor_state != "R":
            return
        # use node IDs from NetworkX to avoid agent objects from NetworkGrid.get_neighbors
        neighbor_nodes = list(self.model.G.neighbors(self.pos))
        for node_id in neighbor_nodes:
            agents = self.model.grid.get_cell_list_contents([node_id])
            if not agents:
                continue
            nb = agents[0]
            if nb.rumor_state == "S" and random.random() < self.model.transmission_prob:
                nb.rumor_state = "I"

    def _demand_response(self):
        if self.rumor_state == "R":
            reduction = self.model.get_reduction_factor(self.income_level)
            self.current_load = self.base_load * (1.0 - reduction)
        else:
            self.current_load = self.base_load

    def _maybe_adopt_solar(self):
        if self.rumor_state != "R" or self.has_solar:
            return
        neighbor_nodes = list(self.model.G.neighbors(self.pos))
        if neighbor_nodes:
            adopted = 0
            for node_id in neighbor_nodes:
                agents = self.model.grid.get_cell_list_contents([node_id])
                if agents and agents[0].has_solar:
                    adopted += 1
            frac_adopted = adopted / len(neighbor_nodes)
        else:
            frac_adopted = 0.0

        income_mult = {"low": 0.6, "medium": 1.0, "high": 1.3}.get(self.income_level, 1.0)
        p = self.innovativeness + self.model.social_influence_weight * frac_adopted
        p *= income_mult
        p += self.model.policy_subsidy
        p = clamp(p, 0.0, 1.0)

        if random.random() < p:
            self.has_solar = True

    def _net_load_after_pv(self):
        if not self.has_solar:
            return self.current_load
        return max(0.0, self.current_load - self.pv_size_kwh_day)

    def step(self):
        self._maybe_progress_rumor()
        self._spread_rumor()
        self._demand_response()
        self._maybe_adopt_solar()
        self.current_load = self._net_load_after_pv()


# ---------------- Model ----------------
class RumorEnergyModel(Model):
    def __init__(
        self,
        df_households: pd.DataFrame,
        avg_degree: int = 3,
        rewiring_prob: float = 0.15,
        transmission_prob: float = 0.25,
        belief_prob: float = 0.4,
        policy_subsidy: float = 0.0,
        social_influence_weight: float = 0.25,
        rng_seed: int | None = 42,
    ):
        super().__init__()
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self.transmission_prob = float(transmission_prob)
        self.belief_prob = float(belief_prob)
        self.policy_subsidy = float(policy_subsidy)
        self.social_influence_weight = float(social_influence_weight)

        # Build network
        self.N = len(df_households)
        self.schedule = RandomActivation(self)
        # small-world network works well for 10–20 households
        self.G = nx.watts_strogatz_graph(n=self.N, k=avg_degree, p=rewiring_prob, seed=rng_seed)
        self.grid = NetworkGrid(self.G)

        # Validate CSV columns
        required = [
            "id", "base_load_kwh", "income", "innovativeness",
            "pv_size_kwh_day", "has_solar", "initial_state"
        ]
        missing = [c for c in required if c not in df_households.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Sort by id so node i == id i
        df_households = df_households.sort_values("id").reset_index(drop=True)

        # Create agents from CSV
        for _, row in df_households.iterrows():
            i = int(row["id"])
            a = Household(
                unique_id=i,
                model=self,
                income_level=str(row["income"]),
                base_load_kwh=float(row["base_load_kwh"]),
                innovativeness=float(row["innovativeness"]),
                pv_size_kwh_day=float(row["pv_size_kwh_day"]),
                has_solar_init=bool(int(row["has_solar"])),
                initial_state=str(row["initial_state"]),
            )
            self.schedule.add(a)
            self.grid.place_agent(a, i)

        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: sum(1 for a in m.schedule.agents if a.rumor_state == "S"),
                "I": lambda m: sum(1 for a in m.schedule.agents if a.rumor_state == "I"),
                "R": lambda m: sum(1 for a in m.schedule.agents if a.rumor_state == "R"),
                "Believers": lambda m: sum(1 for a in m.schedule.agents if a.rumor_state == "R"),
                "SolarAdopters": lambda m: sum(1 for a in m.schedule.agents if a.has_solar),
                "TotalLoad": lambda m: sum(a.current_load for a in m.schedule.agents),
                "AvgLoad": lambda m: np.mean([a.current_load for a in m.schedule.agents]),
            }
        )

    def get_reduction_factor(self, income_level: str) -> float:
       
        if income_level == "low":
            return 0.30
        elif income_level == "medium":
            return 0.20
        else:
            return 0.10

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# --------------- Run & Plot ---------------
def plot_results(df: pd.DataFrame, title_suffix: str = ""):
    plt.figure(figsize=(9,5))
    plt.plot(df.index, df["Believers"], label="Believers")
    plt.plot(df.index, df["SolarAdopters"], label="SolarAdopters")
    plt.plot(df.index, df["TotalLoad"], label="TotalLoad (kWh)")
    plt.xlabel("Time (days)")
    plt.title(f"Believers / Solar / Load {title_suffix}")
    plt.grid(True); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(9,5))
    plt.plot(df.index, df["S"], label="S")
    plt.plot(df.index, df["I"], label="I")
    plt.plot(df.index, df["R"], label="R")
    plt.xlabel("Time (days)")
    plt.ylabel("Agents")
    plt.title(f"S/I/R dynamics {title_suffix}")
    plt.grid(True); plt.legend(); plt.tight_layout()

    plt.show()


def run(csv_path: str, steps: int = 50, subsidy: float = 0.10,
        trans: float = 0.25, belief: float = 0.40, social_w: float = 0.25,
        avg_deg: int = 3, rewire_p: float = 0.15, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    model = RumorEnergyModel(
        df_households=df,
        avg_degree=avg_deg,
        rewiring_prob=rewire_p,
        transmission_prob=trans,
        belief_prob=belief,
        policy_subsidy=subsidy,
        social_influence_weight=social_w,
        rng_seed=seed,
    )
    for _ in range(steps):
        model.step()
    out = model.datacollector.get_model_vars_dataframe()
    out.to_csv("abm_rumor_energy_results_from_csv.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="households.csv", help="Path to households.csv")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--subsidy", type=float, default=0.10)
    parser.add_argument("--trans", type=float, default=0.25)
    parser.add_argument("--belief", type=float, default=0.40)
    parser.add_argument("--social_w", type=float, default=0.25)
    parser.add_argument("--avg_deg", type=int, default=3)
    parser.add_argument("--rewire_p", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run(
        csv_path=args.csv,
        steps=args.steps,
        subsidy=args.subsidy,
        trans=args.trans,
        belief=args.belief,
        social_w=args.social_w,
        avg_deg=args.avg_deg,
        rewire_p=args.rewire_p,
        seed=args.seed
    )

    plot_results(results, title_suffix=f"(N={len(pd.read_csv(args.csv))}, subsidy={args.subsidy:.2f})")


if __name__ == "__main__":
    main()
