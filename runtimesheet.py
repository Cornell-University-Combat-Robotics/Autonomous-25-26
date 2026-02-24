import time
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager


class RuntimeSheet:
    # Used for saving runtimes to a spreadsheet
    def __init__(self, use):
        self.init_time = time.perf_counter()
        self.sheet = []
        self.row = {"Start Time":time.perf_counter()}
        self.use = use

    def log(self, name, value):
        if self.use:
            self.row[name] = value
    
    @contextmanager
    def log_timing(self, name):
        """
        Context manager for timing code blocks.
        
        Usage:
            with rs.log_timing("Operation Name"):
                # code to time
        """
        if self.use:
            start_time = time.perf_counter()
            try:
                yield
            finally:
                elapsed = time.perf_counter() - start_time
                self.log(name, elapsed)
        else:
            yield
    
    def start_iter(self):
        if self.use:
            self.row = {"Start Time":time.perf_counter()}

    def dump(self):
        if self.use:
            self.row["End Time"] = time.perf_counter()
            self.row["Total"] = self.row["End Time"] - self.row["Start Time"]
            self.sheet.append(self.row)
    
    def get_row(self, index):
        if self.use and 0 <= abs(index) < len(self.sheet):
            return self.sheet[index]
        return None

    def save(self, output_name):
        if self.use:
            df = pd.DataFrame(self.sheet)

            # Add a column named Other that is Total minus the sum of all other columns except Start Time, End Time, Total, and FPS10
            df["Other"] = df["Total"] - df.drop(columns=["Start Time", "End Time", "Total", "FPS10"], errors='ignore').sum(axis=1)

            # Limit all floats to 1 decimal place in all columns
            for column in df.columns:
                if column not in ["Start Time", "End Time", "FPS10"]:
                    # Convert column to milliseconds and rename
                    df[column] = df[column] * 1000

            for column in df.columns:
                df[column] = df[column].apply(lambda x: round(x, 1) if isinstance(x, float) else x)

            # Re-oreder columns to have Start Time, End Time, Total, FPS10 then the rest
            column_order = ["Start Time", "End Time", "Total", "FPS10"]
            df = df[column_order + [col for col in df.columns if col not in column_order]]

            # Save to CSV with a timestamp in the filename
            # df.to_csv(output_name, index=True)
            df.to_excel(output_name + ".xlsx", index=True)

            # Make a line graph of all columns except FPS over iterations
            plt.figure(figsize=(10, 5))
            for column in df.columns:
                if column not in ["Start Time", "End Time", "FPS10"]:
                    plt.plot(df.index[1:], df[column][1:], label=column)
            plt.xlabel("Iteration")
            plt.ylabel("Time (ms)")
            plt.title("Runtime Metrics Over Iterations")
            plt.legend()
            plt.savefig(output_name + ".png")
            plt.close()