# HydroSense-Kenya: Scientific Computing Capstone

## Project Overview
HydroSense-Kenya is a Python-based scientific computing model designed to optimize agricultural water-use efficiency. It simulates soil water balance, evaluates uncertainty via Monte Carlo methods, and proposes optimized irrigation schedules using numerical differential equations (Euler/RK4) and custom linear algebra solvers.

## Repository Structure
* `/data`: Contains raw CSVs and the processed, cleaned datasets.
* `/notebooks`: Six incremental Jupyter notebooks documenting the scientific workflow.
* `/src`: Reusable Python modules for data cleaning, numerical methods, simulation, and optimization.
* `/tests`: Pytest suite to validate the integrity of mathematical engines.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the automated tests: `pytest tests/`
4. Launch Jupyter to view the findings: `jupyter notebook`