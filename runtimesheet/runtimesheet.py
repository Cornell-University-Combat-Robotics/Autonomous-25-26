import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager


class RuntimeSheet:
    # Used for saving runtimes to a spreadsheet
    def __init__(self, use):
        self.init_time = time.perf_counter()
        self.sheet = []
        self.row = {"Start Time": time.perf_counter()}
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
            self.row = {"Start Time": time.perf_counter()}

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
        try:
            if self.use:
                print(f"Saving runtime sheet to {output_name}")
                # Ensure the output directory exists and prepend it to the output name
                output_dir = "runtimesheet"
                os.makedirs(output_dir, exist_ok=True)
                if not output_name.startswith(output_dir):
                    output_name = os.path.join(output_dir, output_name)

                df = pd.DataFrame(self.sheet)

                # Add a column named Other that is Total minus the sum of all other columns except Start Time, End Time, Total, and FPS10
                df["Other"] = df["Total"] - df.drop(
                    columns=["Start Time", "End Time", "Total", "FPS10"], errors='ignore').sum(axis=1)

                # Limit all floats to 1 decimal place in all columns
                for column in df.columns:
                    if column not in ["Start Time", "End Time", "FPS10"]:
                        # Convert column to milliseconds and rename
                        df[column] = df[column] * 1000

                for column in df.columns:
                    df[column] = df[column].apply(lambda x: round(
                        x, 1) if isinstance(x, float) else x)

                # Re-oreder columns to have Start Time, End Time, Total, FPS10 then the rest
                column_order = ["Start Time", "End Time", "Total", "FPS10"]
                df = df[column_order +
                        [col for col in df.columns if col not in column_order]]

                # Save to CSV with a timestamp in the filename
                # df.to_csv(output_name, index=True)
                df.to_excel(output_name + ".xlsx", index=True)

                # Make a line graph of all columns except FPS over iterations
                plt.figure(figsize=(10, 5))
                for column in df.columns:
                    if column not in ["Start Time", "End Time", "FPS10"] and df[column].sum() > 1:
                        plt.plot(df.index[1:], df[column][1:], label=column)
                plt.ylim(0, (df["Total"].sum()/len(df)) * 2)
                plt.xlabel("Iteration")
                plt.ylabel("Time (ms)")
                plt.title("Runtime Metrics Over Iterations")
                plt.legend()
                plt.savefig(output_name + ".png")
                # Vector format for infinite zoom
                
                # Set to False for comp, don't want to spend a lot of time on these if we need to restart fast.
                MAKE_COMPLEX_PLOTS = True
                
                if MAKE_COMPLEX_PLOTS:
                    plt.savefig(output_name + ".svg")
                    plt.close()

                    # Make a stacked area graph of the same data
                    plt.figure(figsize=(10, 5))
                    # Get columns to stack (excluding metadata and total)
                    stack_cols = [col for col in df.columns if col not in [
                        "Start Time", "End Time", "Total", "FPS10"] and df[col].sum() > 1]

                    if len(df) > 1 and stack_cols:
                        # Sort columns by average value (lowest to highest for bottom-to-top stacking)
                        sorted_cols = df[stack_cols].mean(
                        ).sort_values().index.tolist()
                        plt.stackplot(df.index[1:], [df[col][1:]
                                    for col in sorted_cols], labels=sorted_cols)
                        plt.ylim(0, (df["Total"].sum()/len(df)) * 2)
                        plt.xlabel("Iteration")
                        plt.ylabel("Time (ms)")
                        plt.title("Stacked Runtime Metrics Over Iterations")
                        plt.legend(loc='upper left')
                        plt.savefig(output_name + "_stacked.png")
                        # Vector format for infinite zoom
                        plt.savefig(output_name + "_stacked.svg")
                    plt.close()

                    # Interactive HTML Export using Plotly
                    try:
                        import plotly.express as px

                        # Prepare a slice of the dataframe (skipping first iteration outlier)
                        pdf = df.iloc[1:].copy()
                        pdf['Iteration'] = pdf.index

                        # Interactive Line Chart
                        # render_mode='webgl' ensures high performance for 4000+ points
                        fig_line = px.line(pdf, x='Iteration', y=stack_cols,
                                        title='Interactive Runtime Metrics (Line View)',
                                        labels={
                                            'value': 'Time (ms)', 'variable': 'Metric'},
                                        render_mode='webgl')
                        fig_line.write_html(output_name + "_interactive.html")

                        # Interactive Stacked Area Chart
                        fig_stack = px.area(pdf, x='Iteration', y=sorted_cols,
                                            title='Interactive Runtime Metrics (Stacked View)',
                                            labels={'value': 'Time (ms)', 'variable': 'Metric'})
                        fig_stack.write_html(output_name + "_interactive_stacked.html")

                    except ImportError:
                        print(
                            "\n[Note] Plotly not found. For interactive HTML graphs, run: pip install plotly")
                        
                else:
                    plt.close()
        except KeyError:
            print("RuntimeSheet failed due to KeyError")
