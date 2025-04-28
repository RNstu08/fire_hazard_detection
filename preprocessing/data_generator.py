import numpy as np
import pandas as pd
import os

def generate_simulated_data(num_samples=10000, fire_probability=0.05, seed=42):
    np.random.seed(seed)
    data = []
    
    for i in range(num_samples):
        fire_event = np.random.rand() < fire_probability
        
        if fire_event:
            temperature = np.random.normal(90, 10)  # Higher temp
            co = np.random.normal(500, 100)         # Higher CO
            hcn = np.random.normal(50, 10)          # Higher HCN
            heat_release = np.random.normal(200, 50)
            smoke_density = np.random.normal(300, 80)
        else:
            temperature = np.random.normal(30, 5)
            co = np.random.normal(5, 2)
            hcn = np.random.normal(1, 0.5)
            heat_release = np.random.normal(10, 5)
            smoke_density = np.random.normal(10, 3)
        
        data.append([temperature, co, hcn, heat_release, smoke_density, int(fire_event)])
    
    df = pd.DataFrame(data, columns=[
        "temperature_C", "CO_ppm", "HCN_ppm", "heat_release_kw", "smoke_density", "fire_label"
    ])
    return df

def save_data(df, filename="simulated_sensor_data.csv"):
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/" + filename, index=False)
    print(f"Data saved at ../data/{filename}")

if __name__ == "__main__":
    df = generate_simulated_data()
    save_data(df)
    print(df.head())
