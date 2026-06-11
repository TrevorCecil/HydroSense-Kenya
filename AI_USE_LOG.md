# AI Assistance Log: HydroSense-Kenya

| Prompt Used | AI Output Summary | Accepted? | Modified? | Validation Method |
| :--- | :--- | :--- | :--- | :--- |
| "Write a Pandas data ingestion script that handles 'NA' and empty strings." | Provided a `load_datasets` function using `na_values=['NA', '']`. | Yes | Minor modifications to file paths. | Printed `df.info()` to verify the NaNs were correctly parsed into the dataframe. |
| "Generate a pytest case for the Bisection method." | Provided a test using the function $x^2 - 4$. | Partly | Adjusted the `atol` tolerance to `1e-4`. | Ran `pytest` locally and verified it passed against the known mathematical root of $x=2$. |
| "Implement the Runge-Kutta 4 (RK4) method for soil moisture." | Provided the standard RK4 logic loop mapping to $S_{t+1}$. | Yes | Edited variable names to match project conventions. | Compared RK4 curve trajectory against Euler method curve to ensure theoretical stability. |
