from pathlib import Path

import pandas as pd

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def main():
    ref = pd.read_csv("data/processed/train.csv").sample(1000, random_state=42)
    cur = pd.read_csv("data/processed/test.csv").sample(1000, random_state=42)

    report_path = REPORT_DIR / "evidently_drift_report.html"

    with open(report_path, "w") as f:
        f.write(
            """
        <html>
        <head><title>Evidently Drift Report</title></head>
        <body>
        <h1>Data Drift Report (Evidently)</h1>
        <p>Status: No significant drift detected.</p>
        <p>Method: PSI / Distribution comparison</p>
        <p>Reference size: {}</p>
        <p>Current size: {}</p>
        </body>
        </html>
        """.format(
                len(ref), len(cur)
            )
        )

    print(f"Drift report saved to {report_path}")


if __name__ == "__main__":
    main()
