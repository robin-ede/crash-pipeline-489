"""
PDF Report Generator for Chicago Crash ETL Pipeline
Generates comprehensive PDF reports using ReportLab.
"""
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_run_history_pdf(summary_data: dict, run_history: list) -> BytesIO:
    """
    Generate a comprehensive PDF report for pipeline run history.

    Args:
        summary_data: Dictionary with summary metrics (total_runs, latest_corrid, etc.)
        run_history: List of run history records

    Returns:
        BytesIO object containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#ff7f0e'),
        spaceAfter=12,
        spaceBefore=12
    )

    # Title
    title = Paragraph("Chicago Crash ETL Pipeline<br/>Run History Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    # Report metadata
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = Paragraph(f"<b>Generated:</b> {report_time}", styles['Normal'])
    story.append(metadata)
    story.append(Spacer(1, 0.3 * inch))

    # Section 1: Pipeline Summary
    story.append(Paragraph("Pipeline Summary", heading_style))

    # Format gold_row_count safely
    gold_rows = summary_data.get('gold_row_count', 0)
    if isinstance(gold_rows, (int, float)):
        gold_rows_str = f"{int(gold_rows):,}"
    else:
        gold_rows_str = str(gold_rows)

    summary_data_table = [
        ["Metric", "Value"],
        ["Total Pipeline Runs", str(summary_data.get('total_runs', 0))],
        ["Latest Correlation ID", str(summary_data.get('latest_corrid', 'N/A'))],
        ["Gold Database Row Count", gold_rows_str],
        ["Latest Data Date", str(summary_data.get('latest_data_date', 'N/A'))],
        ["Last Run Timestamp", str(summary_data.get('last_run_timestamp', 'N/A'))]
    ]

    summary_table = Table(summary_data_table, colWidths=[3 * inch, 3.5 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4 * inch))

    # Section 2: Run History
    if run_history and len(run_history) > 0:
        story.append(Paragraph(f"Run History (Last {len(run_history)} Runs)", heading_style))

        # Prepare run history table
        history_data = [["CorrID", "Mode", "Window", "Rows", "Status"]]

        for run in run_history[:20]:  # Limit to most recent 20
            corrid = run.get('corrid', 'N/A')[:15] + "..."  # Truncate for space
            mode = run.get('mode', 'N/A')
            window = run.get('window', 'N/A')

            # Handle rows - could be int or string
            rows_value = run.get('rows', 'N/A')
            if isinstance(rows_value, (int, float)):
                rows = f"{int(rows_value):,}"
            else:
                rows = str(rows_value)

            status = run.get('status', 'N/A')

            history_data.append([corrid, mode, window, rows, status])

        history_table = Table(history_data, colWidths=[1.8 * inch, 1 * inch, 1.5 * inch, 0.8 * inch, 1 * inch])
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(history_table)
        story.append(Spacer(1, 0.4 * inch))
    else:
        story.append(Paragraph("No run history available.", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

    # Section 3: System Information
    story.append(PageBreak())
    story.append(Paragraph("System Information", heading_style))

    system_info_text = f"""
    <b>Pipeline Architecture:</b><br/>
    • Extractor: Pulls data from Socrata API (crashes, vehicles, people)<br/>
    • Transformer: Merges datasets into unified CSV format<br/>
    • Cleaner: Performs data cleaning and loads into Gold DuckDB<br/>
    <br/>
    <b>Storage:</b><br/>
    • Object Store: MinIO (raw-data, transform-data buckets)<br/>
    • Warehouse: DuckDB (gold.duckdb)<br/>
    <br/>
    <b>Orchestration:</b><br/>
    • Message Queue: RabbitMQ<br/>
    • API: FastAPI (port 8000)<br/>
    • Dashboard: Streamlit (port 8501)<br/>
    • Scheduler: APScheduler (cron-based automation)<br/>
    """

    system_info = Paragraph(system_info_text, styles['Normal'])
    story.append(system_info)
    story.append(Spacer(1, 0.3 * inch))

    # Section 4: Notes
    story.append(Paragraph("Notes", heading_style))

    notes_text = """
    This report provides a snapshot of the Chicago Crash ETL pipeline's execution history.
    For real-time monitoring, access the Streamlit dashboard or query the Gold database directly.
    <br/><br/>
    <b>Data Sources:</b><br/>
    • Chicago Data Portal: https://data.cityofchicago.org<br/>
    • Crashes Dataset ID: 85ca-t3if<br/>
    • Vehicles Dataset ID: 68nd-jvt3<br/>
    • People Dataset ID: u6pd-qa9d<br/>
    """

    notes = Paragraph(notes_text, styles['Normal'])
    story.append(notes)

    # Footer
    story.append(Spacer(1, 0.5 * inch))
    footer = Paragraph(
        f"<i>Chicago Crash ETL Pipeline Report • Generated: {report_time}</i>",
        styles['Normal']
    )
    story.append(footer)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_simple_summary_pdf(summary_data: dict) -> BytesIO:
    """
    Generate a simplified PDF report with just summary metrics.
    Useful for quick reports without full run history.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Chicago Crash ETL Pipeline<br/>Summary Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.3 * inch))

    # Summary
    summary_text = f"""
    <b>Total Pipeline Runs:</b> {summary_data.get('total_runs', 0)}<br/>
    <b>Latest Correlation ID:</b> {summary_data.get('latest_corrid', 'N/A')}<br/>
    <b>Gold Database Row Count:</b> {summary_data.get('gold_row_count', 0):,}<br/>
    <b>Latest Data Date:</b> {summary_data.get('latest_data_date', 'N/A')}<br/>
    <b>Last Run Timestamp:</b> {summary_data.get('last_run_timestamp', 'N/A')}<br/>
    <br/>
    <i>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>
    """

    summary = Paragraph(summary_text, styles['Normal'])
    story.append(summary)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
