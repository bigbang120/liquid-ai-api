from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

app = FastAPI(title="Liquid AI Exploratory Deviation API")

COLUMN_MAP = {
    "heart_rate": ["hr", "heart rate", "heartrate", "heart_rate"],
    "spo2": ["spo2", "sp02", "oxygen", "blood oxygen", "o2"],
    "systolic_bp": ["sys", "systolic", "systolic bp", "sbp"],
    "diastolic_bp": ["dia", "diastolic", "diastolic bp", "dbp"],
}

def map_columns(df):
    mapped = {}
    for standard, variations in COLUMN_MAP.items():
        for col in df.columns:
            if col.strip().lower() in variations:
                mapped[standard] = col
    return mapped

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    column_mapping = map_columns(df)

    baselines = {}
    volatility = {}
    deviation_rows = []

    # --- Baseline + Personal Volatility (IQR) ---
    for signal, col in column_mapping.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        clean_series = df[col].dropna()

        if clean_series.empty:
            continue

        baseline = clean_series.median()

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            iqr = 1e-6

        baselines[signal] = baseline
        volatility[signal] = iqr

    # --- Deviation Detection ---
    for idx, row in df.iterrows():
        triggered = []

        for signal, col in column_mapping.items():
            if signal not in baselines:
                continue

            value = row[col]
            baseline = baselines[signal]

            if pd.isna(value):
                continue

            if signal == "heart_rate":
                if abs(value - baseline) / baseline > 0.15:
                    triggered.append("heart_rate")

            elif signal == "spo2":
                if baseline - value > 2:
                    triggered.append("spo2")

            elif signal == "systolic_bp":
                if abs(value - baseline) > 10:
                    triggered.append("systolic_bp")

            elif signal == "diastolic_bp":
                if abs(value - baseline) > 5:
                    triggered.append("diastolic_bp")

        if triggered:
            deviation_rows.append({
                "row": idx,
                "signals": ", ".join(triggered),
                "severity": len(triggered)
            })

    # --- Generate PDF ---
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Liquid AI Exploratory Deviation Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    # Baseline Table with Volatility
    baseline_table_data = [["Signal", "Median Baseline", "Volatility (IQR)"]]

    for signal in baselines:
        baseline_table_data.append([
            signal,
            round(baselines[signal], 2),
            round(volatility[signal], 2)
        ])

    baseline_table = Table(baseline_table_data)
    baseline_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ])

    elements.append(Paragraph("Baselines:", styles["Heading2"]))
    elements.append(baseline_table)
    elements.append(Spacer(1, 20))

    # Deviation Table
    deviation_table_data = [["Row", "Signals Triggered", "Severity"]]

    for row in deviation_rows:
        deviation_table_data.append([row["row"], row["signals"], row["severity"]])

    deviation_table = Table(deviation_table_data)
    deviation_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ])

    elements.append(Paragraph("Deviation Events:", styles["Heading2"]))
    elements.append(deviation_table)
    elements.append(Spacer(1, 20))

    total_rows = len(df)
    deviation_count = len(deviation_rows)
    max_severity = max([row["severity"] for row in deviation_rows], default=0)
    multi_signal = len([row for row in deviation_rows if row["severity"] >= 2])

    summary_data = [
        ["Total Rows", total_rows],
        ["Deviation Rows", deviation_count],
        ["Max Severity", max_severity],
        ["Multi-signal Instability (>=2)", multi_signal]
    ]

    summary_table = Table(summary_data)
    summary_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
    ])

    elements.append(Paragraph("Summary:", styles["Heading2"]))
    elements.append(summary_table)
    elements.append(Spacer(1, 30))

    elements.append(Paragraph(
        "For research / exploratory use only. Not a medical device. Not diagnostic. No clinical claims.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=liquid_ai_report.pdf"}
    )
