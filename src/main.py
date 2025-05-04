from src.etl.pipeline import Pipeline
from src.dashboard import dashboard
from dotenv import load_dotenv

import pandas as pd

load_dotenv()


def parse(file_path, ticker):
    etl = Pipeline()
    parsed: pd.DataFrame = etl.run(
        file_path,
        output_format="df",
        ticker=ticker,
        include_financial_data=True
        )
    print(parsed)
    # Save the DataFrame to a CSV file
    parsed.to_csv("output.csv", index=False)
    parsed.to_excel("output.xlsx", index=False)
    return parsed


if __name__ == "__main__":
    df = parse("data/BRTZ-43-101_Technical_Report_Merged_22-02-09_Final__3_.pdf", "GMIN.TO")
    # df = parse("data/MadsenPFS-NI43-101-Final-20250218.pdf", "WRLG.V")
    dashboard(df, title="G Mining Tocantinzinho")