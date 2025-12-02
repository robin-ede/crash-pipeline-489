"""
Chicago Crash ETL Dashboard - Streamlit Application
Interactive dashboard for managing and visualizing the crash data pipeline.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import time

# =============================
# Configuration
# =============================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_ARTIFACT_PATH = "artifacts/pipeline_calibrated.pkl"
GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "/data/gold/gold.duckdb")

# =============================
# Prometheus Metrics (cached to avoid re-registration)
# =============================
@st.cache_resource
def get_metrics():
    """Create and return Prometheus metrics (cached to avoid duplicates)."""
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    metrics = {
        'page_views': Counter('dashboard_page_views_total', 'Total page views', ['page']),
        'predictions_made': Counter('dashboard_predictions_total', 'Total predictions made via dashboard'),
        'prediction_latency': Histogram('dashboard_prediction_latency_seconds', 'Prediction latency in dashboard', buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]),
        'query_duration': Histogram('dashboard_query_duration_seconds', 'Database query duration', buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]),
        'model_accuracy': Gauge('dashboard_model_accuracy', 'Current model accuracy'),
        'model_precision': Gauge('dashboard_model_precision', 'Current model precision'),
        'model_recall': Gauge('dashboard_model_recall', 'Current model recall'),
        'ml_training_duration': Histogram('dashboard_ml_training_duration_seconds', 'ML training duration', buckets=[1, 5, 10, 30, 60, 300, 600])
    }

    # Start metrics server
    try:
        start_http_server(8003)
    except OSError:
        pass  # Already running

    return metrics

# Get metrics (will be cached)
_metrics = get_metrics()
page_views = _metrics['page_views']
predictions_made = _metrics['predictions_made']
prediction_latency = _metrics['prediction_latency']
query_duration = _metrics['query_duration']
model_accuracy = _metrics['model_accuracy']
model_precision = _metrics['model_precision']
model_recall = _metrics['model_recall']
ml_training_duration = _metrics['ml_training_duration']

# =============================
# Model & Database Helpers
# =============================
@st.cache_resource
def load_model():
    """
    Load the ML model artifact with Streamlit caching.
    Model is loaded once per session and reused.
    """
    import joblib
    from pathlib import Path

    model_path = Path(MODEL_ARTIFACT_PATH)

    if not model_path.exists():
        st.error(f"❌ Model file not found at `{MODEL_ARTIFACT_PATH}`")
        st.error("Please ensure the model artifact exists in the dashboard/artifacts folder.")
        st.stop()

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.error("The model file may be corrupted or incompatible.")
        st.stop()


@st.cache_resource
def get_duckdb_connection():
    """
    Get a cached DuckDB connection.
    Connection is reused for all queries in the session.
    """
    import duckdb
    from pathlib import Path

    db_path = Path(GOLD_DB_PATH)

    if not db_path.exists():
        st.error(f"❌ Gold database not found at `{GOLD_DB_PATH}`")
        st.error("Please ensure the Gold DuckDB file exists.")
        st.stop()

    try:
        # Use read_only mode with WAL for concurrent access
        con = duckdb.connect(str(db_path), read_only=True, config={'access_mode': 'READ_ONLY'})
        return con
    except Exception as e:
        st.error(f"❌ Failed to connect to Gold database: {e}")
        st.stop()


def query_gold_db(sql):
    """Execute SQL query on Gold database using cached connection."""
    try:
        start_time = time.time()
        con = get_duckdb_connection()
        result = con.execute(sql).fetchdf()
        query_duration.observe(time.time() - start_time)
        return result
    except Exception as e:
        st.error(f"❌ Query failed: {e}")
        return None

# Default columns from streaming.json (all lowercase)
DEFAULT_CRASHES_COLS = [
    "crash_record_id", "crash_date", "posted_speed_limit", "traffic_control_device",
    "device_condition", "weather_condition", "lighting_condition", "first_crash_type",
    "trafficway_type", "lane_cnt", "alignment", "roadway_surface_cond", "road_defect",
    "intersection_related_i", "hit_and_run_i", "work_zone_i", "work_zone_type",
    "workers_present_i", "num_units", "crash_hour", "crash_day_of_week", "crash_month",
    "beat_of_occurrence", "latitude", "longitude", "injuries_total", "injuries_fatal",
    "injuries_incapacitating", "injuries_non_incapacitating", "injuries_reported_not_evident"
]

DEFAULT_VEHICLES_COLS = [
    "crash_record_id", "unit_no", "vehicle_id", "unit_type", "vehicle_type",
    "vehicle_use", "vehicle_year", "travel_direction", "maneuver", "towed_i",
    "exceed_speed_limit_i", "occupant_cnt", "num_passengers"
]

DEFAULT_PEOPLE_COLS = [
    "crash_record_id", "person_id", "person_type", "age", "sex",
    "safety_equipment", "airbag_deployed", "ejection"
]

st.set_page_config(
    page_title="Chicago Crash ETL Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ff7f0e;
        margin-top: 1.5rem;
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Helper Functions
# =============================
def call_api(endpoint, method="GET", data=None, params=None):
    """Helper function to call API endpoints."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, json=data, timeout=30)
        else:
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to API at {API_BASE_URL}. Is the API server running?")
        return None
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_health_status():
    """Get health status of all services."""
    return call_api("/api/health")

@st.cache_data(ttl=300)
def get_schema_columns(dataset):
    """Get column schema for a dataset."""
    return call_api("/api/schema/columns", params={"dataset": dataset})

@st.cache_data(ttl=30)
def get_gold_status():
    """Get Gold database status."""
    return call_api("/api/gold/status")

@st.cache_data(ttl=10)
def browse_minio_folders(bucket):
    """Browse folders in a MinIO bucket."""
    return call_api("/api/minio/browse", params={"bucket": bucket})

def get_gold_data(query):
    """Execute SQL query on Gold database."""
    return call_api("/api/gold/query", params={"sql": query})

# =============================
# Tab 1: Home
# =============================
def render_home_tab():
    st.markdown('<div class="main-header">🌆 Chicago Crash ETL Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Label Overview Card
    st.markdown('<div class="sub-header">🎯 Injury Prediction Pipeline</div>', unsafe_allow_html=True)

    overview_col1, overview_col2 = st.columns([2, 1])

    with overview_col1:
        st.markdown("""
        <div class="info-card">
        <h3>🚨 Injury Outcome</h3>
        <p><strong>Label predicted:</strong> <code>injury</code> • <strong>Type:</strong> binary • <strong>Positive class:</strong> 1 (injury occurred)</p>

        <p><strong>Pipeline:</strong><br/>
        We built a model to predict whether a crash resulted in injuries using crash context signals like speed limits, weather conditions, time patterns, location, and vehicle/people characteristics.</p>

        <p><strong>Key features (why they help):</strong></p>
        <ul>
        <li><code>posted_speed_limit</code> — Higher speeds correlate with injury severity</li>
        <li><code>weather_condition</code> — Adverse weather increases injury risk</li>
        <li><code>lighting_condition</code> — Poor visibility (dark, no lights) raises severity</li>
        <li><code>hour_bin</code> & <code>is_weekend</code> — Time patterns reveal behavioral differences</li>
        <li><code>veh_truck_i</code> & <code>veh_mc_i</code> — Vehicle type impacts injury outcomes</li>
        <li><code>ppl_age_mean</code> — Age demographics affect vulnerability</li>
        <li><code>intersection_related_i</code> — Intersection crashes have distinct injury patterns</li>
        </ul>

        <p><strong>Source columns (subset):</strong></p>
        <ul>
        <li><code>crashes</code>: crash_date, posted_speed_limit, weather_condition, lighting_condition, first_crash_type, latitude, longitude, injuries_total, injuries_fatal, injuries_incapacitating, injuries_non_incapacitating, injuries_reported_not_evident</li>
        <li><code>vehicles</code>: vehicle_type (for truck/motorcycle flags)</li>
        <li><code>people</code>: age (for age statistics)</li>
        </ul>

        <p><strong>Class imbalance:</strong></p>
        <ul>
        <li>Any injury represents a more balanced target (expected ~30-40% of crashes)</li>
        <li>Handling: Monitor class distribution. Use <code>class_weight='balanced'</code> or adjust classification threshold based on precision-recall tradeoff</li>
        </ul>

        <p><strong>Data grain & filters:</strong></p>
        <ul>
        <li>One row = <strong>one crash</strong> (crash-level grain)</li>
        <li>Window: Configurable via streaming (since_days) or backfill (date range)</li>
        <li>Filters: Dropped rows with invalid coordinates (outside Chicago bounds)</li>
        </ul>

        <p><strong>Leakage/caveats:</strong></p>
        <ul>
        <li>Source injury columns (<code>injuries_total</code>, <code>injuries_fatal</code>, <code>injuries_incapacitating</code>, <code>injuries_non_incapacitating</code>, <code>injuries_reported_not_evident</code>) are <strong>dropped after creating the target</strong> to prevent leakage</li>
        <li>ID columns (<code>vehicle_id</code>, <code>person_id</code>) are excluded from features</li>
        <li>Only use pre-crash attributes (time, location, environment, vehicle type) — never post-crash outcomes</li>
        </ul>

        <p><strong>Gold table:</strong> <code>gold.main.crashes</code></p>
        </div>
        """, unsafe_allow_html=True)

    with overview_col2:
        st.info("**Quick Links**")
        st.markdown("- 📊 Jump to **EDA** for visualizations")
        st.markdown("- 📑 Jump to **Reports** for pipeline stats")
        st.markdown("- 🔍 Use **Data Fetcher** to pull new data")
        st.markdown("- 🧰 **Data Management** for cleanup")

    st.markdown("---")

    # Container Health
    st.markdown('<div class="sub-header">🏥 Container Health Status</div>', unsafe_allow_html=True)

    health_data = get_health_status()

    if health_data:
        health_cols = st.columns(5)
        services = health_data.get("services", [])

        for idx, service in enumerate(services):
            with health_cols[idx]:
                service_name = service.get("service", "Unknown")
                status = service.get("status", "unknown")
                message = service.get("message", "")

                # Determine icon and status text based on status
                if status == "healthy":
                    icon = "✅"
                    status_text = "Ready"  # Changed from "Active" - clearer for idle workers
                    status_class = "status-healthy"
                elif status == "idle":
                    icon = "🟡"
                    status_text = "Stopped"  # Changed from "Idle" - service not connected
                    status_class = "status-healthy"
                elif status == "unknown":
                    icon = "⚪"
                    status_text = "Unknown"
                    status_class = "status-healthy"
                else:  # unhealthy
                    icon = "❌"
                    status_text = "Error"
                    status_class = "status-unhealthy"

                st.markdown(f"""
                <div class="metric-card">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: bold; margin-top: 0.5rem;">{service_name}</div>
                <div class="{status_class}">{status_text}</div>
                <div style="font-size: 0.75rem; color: #666; margin-top: 0.25rem;">{message[:40]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(f"Last checked: {health_data.get('timestamp', 'N/A')}")
    else:
        st.warning("Unable to fetch health status from API.")

# =============================
# Tab 2: Data Management
# =============================
def render_data_management_tab():
    st.markdown('<div class="sub-header">🧰 Data Management</div>', unsafe_allow_html=True)

    # MinIO Management
    st.markdown("### 📦 MinIO Browser & Delete")

    # Refresh button for MinIO browser
    browse_refresh_col1, browse_refresh_col2 = st.columns([6, 1])
    with browse_refresh_col2:
        if st.button("🔄 Refresh", key="refresh_minio"):
            st.cache_data.clear()
            st.rerun()

    minio_action = st.radio("Select action:", ["Browse & Delete Folder", "Wipe Bucket Contents"], horizontal=True)

    if minio_action == "Browse & Delete Folder":
        # Select bucket
        bucket = st.selectbox("Select Bucket", ["raw-data", "transform-data"], key="minio_folder_bucket")

        # Browse folders in selected bucket
        with st.spinner(f"Loading folders from {bucket}..."):
            browse_data = browse_minio_folders(bucket)

        if browse_data and browse_data.get("folders"):
            folders = browse_data["folders"]
            st.info(f"Found **{browse_data['total_folders']}** folders in **{bucket}**")

            # Create a table showing all folders
            st.markdown("#### 📂 Available Folders")

            # Convert to DataFrame for display
            folders_df = pd.DataFrame(folders)
            folders_df = folders_df[["prefix", "count", "size_mb", "last_modified"]]
            folders_df.columns = ["Folder Path", "Object Count", "Size (MB)", "Last Modified"]

            # Format last_modified to be more readable
            if "Last Modified" in folders_df.columns:
                folders_df["Last Modified"] = folders_df["Last Modified"].apply(
                    lambda x: x[:19].replace("T", " ") if x else "N/A"
                )

            # Display the table
            st.dataframe(folders_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Select folder to delete
            st.markdown("#### 🗑️ Delete Folder")

            # Create selectbox with folder options
            folder_options = [f["prefix"] for f in folders]
            selected_folder = st.selectbox(
                "Select folder to delete",
                options=[""] + folder_options,  # Add empty option as default
                format_func=lambda x: "-- Select a folder --" if x == "" else x,
                key="selected_folder_to_delete"
            )

            if selected_folder:
                # Find selected folder data
                folder_data = next((f for f in folders if f["prefix"] == selected_folder), None)

                if folder_data:
                    # Show folder details
                    st.info(f"""
                    **Folder:** `{folder_data['prefix']}`
                    **Objects:** {folder_data['count']}
                    **Size:** {folder_data['size_mb']} MB
                    **Last Modified:** {folder_data['last_modified'][:19].replace('T', ' ') if folder_data['last_modified'] else 'N/A'}
                    """)

                    # Preview button
                    if st.button("🔍 Preview Objects", key="preview_objects"):
                        with st.spinner("Loading object list..."):
                            preview_data = call_api("/api/minio/preview", params={"bucket": bucket, "prefix": selected_folder})
                            if preview_data and preview_data.get("objects"):
                                st.success(f"Found **{preview_data['total_objects']}** objects")

                                # Show first 20 objects
                                preview_objects = preview_data["objects"][:20]
                                preview_df = pd.DataFrame(preview_objects)

                                if not preview_df.empty:
                                    preview_df = preview_df[["key", "size", "last_modified"]]
                                    preview_df.columns = ["Object Key", "Size (bytes)", "Last Modified"]
                                    preview_df["Last Modified"] = preview_df["Last Modified"].apply(
                                        lambda x: x[:19].replace("T", " ") if x else "N/A"
                                    )
                                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

                                    if len(preview_data["objects"]) > 20:
                                        st.caption(f"Showing first 20 of {preview_data['total_objects']} objects")

                    # Confirmation and delete
                    confirm_folder = st.checkbox(
                        f"✅ I confirm deletion of **{folder_data['count']} objects** from `{selected_folder}`",
                        key="confirm_folder"
                    )

                    if st.button("🗑️ Delete Folder", disabled=not confirm_folder, key="delete_folder_btn"):
                        with st.spinner("Deleting..."):
                            result = call_api("/api/minio/folder", method="DELETE", data={"bucket": bucket, "prefix": selected_folder})
                            if result:
                                st.success(f"✅ Deleted {result.get('deleted_count', 0)} objects from {bucket}/{selected_folder}")
                                # Clear all cached data to force refresh
                                browse_minio_folders.clear()
                                get_gold_status.clear()
                                st.rerun()

        else:
            st.warning(f"No folders found in **{bucket}** bucket. The bucket may be empty.")

    else:  # Delete Entire Bucket
        bucket = st.selectbox("Select Bucket", ["raw-data", "transform-data"], key="minio_bucket_delete")

        st.warning(f"⚠️ This will permanently wipe all contents from the **{bucket}** bucket!")

        # Show bucket stats
        with st.spinner("Loading bucket stats..."):
            browse_data = browse_minio_folders(bucket)
            if browse_data:
                st.info(f"""
                **Bucket:** `{bucket}`
                **Total Folders:** {browse_data.get('total_folders', 0)}
                **Total Objects:** {sum(f['count'] for f in browse_data.get('folders', []))}
                """)

        confirm_bucket = st.checkbox("✅ I confirm wiping all contents from this bucket", key="confirm_bucket")

        if st.button("🗑️ Wipe Bucket", disabled=not confirm_bucket, key="delete_bucket_btn"):
            with st.spinner("Wiping bucket contents..."):
                result = call_api("/api/minio/bucket", method="DELETE", data={"bucket": bucket})
                if result:
                    st.success(f"✅ {result.get('message', 'Bucket wiped successfully')}")
                    # Clear cached data to force refresh
                    browse_minio_folders.clear()
                    get_gold_status.clear()
                    st.rerun()

    st.markdown("---")

    # Gold Admin
    st.markdown("### 🏆 Gold Admin (DuckDB)")

    gold_status = get_gold_status()

    if gold_status and gold_status.get("exists"):
        st.info(f"""
        **Database Path:** `{gold_status['path']}`
        **File Size:** {gold_status['file_size_mb']} MB
        **Total Tables:** {gold_status['total_tables']}
        **Total Rows:** {gold_status['total_rows']}
        """)

        if gold_status.get("tables"):
            st.dataframe(pd.DataFrame(gold_status["tables"]), use_container_width=True)
    else:
        st.warning("Gold database does not exist yet.")

    confirm_wipe = st.checkbox("✅ I confirm wiping the entire Gold database", key="confirm_wipe")

    if st.button("🗑️ Wipe Gold DB (ENTIRE FILE)", disabled=not confirm_wipe):
        with st.spinner("Wiping database..."):
            result = call_api("/api/gold/wipe", method="DELETE")
            if result:
                st.success(f"✅ {result.get('message', 'Database wiped successfully')}")
                # Clear cached data to force refresh
                get_gold_status.clear()
                st.rerun()

    st.markdown("---")

    # Quick Peek
    st.markdown("### 👀 Quick Peek (Gold — sanity view)")

    peek_col1, peek_col2 = st.columns([3, 1])
    with peek_col1:
        peek_columns = st.text_input("Columns (comma-separated, leave empty for auto-select first 8)", key="peek_cols")
    with peek_col2:
        peek_limit = st.slider("Rows (limit)", 10, 200, 50, key="peek_limit")

    if st.button("👁️ Preview", key="preview_peek"):
        with st.spinner("Fetching data..."):
            params = {"table": '"gold"."main"."crashes"', "limit": peek_limit}
            if peek_columns:
                params["columns"] = peek_columns

            peek_data = call_api("/api/gold/peek", params=params)
            if peek_data:
                df = pd.DataFrame(peek_data["data"])
                st.dataframe(df, use_container_width=True)
                st.caption(f"Showing {len(df)} rows from {peek_data['table']}")

# =============================
# Tab 3: Data Fetcher
# =============================
def render_data_fetcher_tab():
    st.markdown('<div class="sub-header">🔍 Data Fetcher</div>', unsafe_allow_html=True)

    # Subtabs for Streaming and Backfill
    fetch_subtabs = st.tabs(["📡 Streaming", "🕰️ Backfill"])

    # ------- Streaming Subtab -------
    with fetch_subtabs[0]:
        st.markdown("#### Fetch recent crash data (last N days)")

        # Generate corrid
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        corrid_streaming = f"{timestamp}_streaming"
        st.text_input("CorrID (auto-generated)", corrid_streaming, disabled=True, key="corrid_stream")

        since_days = st.number_input("Since days", min_value=1, max_value=3650, value=30, key="since_days")

        # Crashes columns (always fetched)
        st.markdown("##### Crashes: Columns to fetch")
        crashes_cols = get_schema_columns("crashes")
        if crashes_cols and crashes_cols.get("columns"):
            crashes_col_names = [c["name"].lower() for c in crashes_cols["columns"]]
            # Use defaults from streaming.json, only include those that exist in API
            default_crashes = [col for col in DEFAULT_CRASHES_COLS if col in crashes_col_names]
            selected_crashes = st.multiselect(
                "Crashes columns",
                crashes_col_names,
                default=default_crashes,
                key="crashes_cols_stream"
            )
        else:
            selected_crashes = []
            st.warning("Could not load crashes schema.")

        # Enrichment: Vehicles
        include_vehicles = st.checkbox("✅ Include Vehicles", value=True, key="include_veh_stream")
        if include_vehicles:
            vehicles_cols = get_schema_columns("vehicles")
            if vehicles_cols and vehicles_cols.get("columns"):
                vehicles_col_names = [c["name"].lower() for c in vehicles_cols["columns"]]
                select_all_veh = st.checkbox("Select all vehicle columns", key="select_all_veh_stream")
                if select_all_veh:
                    selected_vehicles = vehicles_col_names
                else:
                    # Use defaults from streaming.json
                    default_vehicles = [col for col in DEFAULT_VEHICLES_COLS if col in vehicles_col_names]
                    selected_vehicles = st.multiselect(
                        "Vehicles: columns to be fetched",
                        vehicles_col_names,
                        default=default_vehicles,
                        key="vehicles_cols_stream"
                    )
            else:
                selected_vehicles = []
        else:
            selected_vehicles = []

        # Enrichment: People
        include_people = st.checkbox("✅ Include People", value=True, key="include_ppl_stream")
        if include_people:
            people_cols = get_schema_columns("people")
            if people_cols and people_cols.get("columns"):
                people_col_names = [c["name"].lower() for c in people_cols["columns"]]
                select_all_ppl = st.checkbox("Select all people columns", key="select_all_ppl_stream")
                if select_all_ppl:
                    selected_people = people_col_names
                else:
                    # Use defaults from streaming.json
                    default_people = [col for col in DEFAULT_PEOPLE_COLS if col in people_col_names]
                    selected_people = st.multiselect(
                        "People: columns to be fetched",
                        people_col_names,
                        default=default_people,
                        key="people_cols_stream"
                    )
            else:
                selected_people = []
        else:
            selected_people = []

        # Preview JSON
        with st.expander("🔍 Preview JSON Request"):
            preview_json_stream = {
                "mode": "streaming",
                "since_days": since_days,
                "crashes_columns": selected_crashes,
                "vehicles_columns": selected_vehicles if include_vehicles else [],
                "people_columns": selected_people if include_people else [],
                "include_vehicles": include_vehicles,
                "include_people": include_people
            }
            st.json(preview_json_stream)

        # Buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("📤 Publish to RabbitMQ", key="publish_stream"):
                with st.spinner("Publishing job..."):
                    result = call_api("/api/fetch/publish", method="POST", data=preview_json_stream)
                    if result and result.get("status") == "success":
                        st.success(f"✅ Job queued! CorrID: **{result['corrid']}**")
                        st.json(result)
                    else:
                        st.error("Failed to publish job.")

        with button_col2:
            if st.button("🔄 Reset form", key="reset_stream"):
                st.rerun()

    # ------- Backfill Subtab -------
    with fetch_subtabs[1]:
        st.markdown("#### Fetch historical crash data (date range)")

        # Generate corrid
        corrid_backfill = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_backfill"
        st.text_input("CorrID (auto-generated)", corrid_backfill, disabled=True, key="corrid_backfill")

        # Date range
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=7), key="start_date")
            start_time = st.time_input("Start time", value=datetime.strptime("00:00:00", "%H:%M:%S").time(), key="start_time")
        with date_col2:
            end_date = st.date_input("End date", value=datetime.now(), key="end_date")
            end_time = st.time_input("End time", value=datetime.strptime("23:59:59", "%H:%M:%S").time(), key="end_time")

        # Crashes columns
        st.markdown("##### Crashes: Columns to fetch")
        crashes_cols_bf = get_schema_columns("crashes")
        if crashes_cols_bf and crashes_cols_bf.get("columns"):
            crashes_col_names_bf = [c["name"].lower() for c in crashes_cols_bf["columns"]]
            # Use defaults from streaming.json
            default_crashes_bf = [col for col in DEFAULT_CRASHES_COLS if col in crashes_col_names_bf]
            selected_crashes_bf = st.multiselect(
                "Crashes columns",
                crashes_col_names_bf,
                default=default_crashes_bf,
                key="crashes_cols_backfill"
            )
        else:
            selected_crashes_bf = []

        # Vehicles enrichment
        include_vehicles_bf = st.checkbox("✅ Include Vehicles", value=True, key="include_veh_backfill")
        if include_vehicles_bf:
            vehicles_cols_bf = get_schema_columns("vehicles")
            if vehicles_cols_bf and vehicles_cols_bf.get("columns"):
                vehicles_col_names_bf = [c["name"].lower() for c in vehicles_cols_bf["columns"]]
                select_all_veh_bf = st.checkbox("Select all vehicle columns", key="select_all_veh_backfill")
                if select_all_veh_bf:
                    selected_vehicles_bf = vehicles_col_names_bf
                else:
                    # Use defaults from streaming.json
                    default_vehicles_bf = [col for col in DEFAULT_VEHICLES_COLS if col in vehicles_col_names_bf]
                    selected_vehicles_bf = st.multiselect(
                        "Vehicles: columns to be fetched",
                        vehicles_col_names_bf,
                        default=default_vehicles_bf,
                        key="vehicles_cols_backfill"
                    )
            else:
                selected_vehicles_bf = []
        else:
            selected_vehicles_bf = []

        # People enrichment
        include_people_bf = st.checkbox("✅ Include People", value=True, key="include_ppl_backfill")
        if include_people_bf:
            people_cols_bf = get_schema_columns("people")
            if people_cols_bf and people_cols_bf.get("columns"):
                people_col_names_bf = [c["name"].lower() for c in people_cols_bf["columns"]]
                select_all_ppl_bf = st.checkbox("Select all people columns", key="select_all_ppl_backfill")
                if select_all_ppl_bf:
                    selected_people_bf = people_col_names_bf
                else:
                    # Use defaults from streaming.json
                    default_people_bf = [col for col in DEFAULT_PEOPLE_COLS if col in people_col_names_bf]
                    selected_people_bf = st.multiselect(
                        "People: columns to be fetched",
                        people_col_names_bf,
                        default=default_people_bf,
                        key="people_cols_backfill"
                    )
            else:
                selected_people_bf = []
        else:
            selected_people_bf = []

        # Preview JSON
        with st.expander("🔍 Preview JSON Request"):
            preview_json_backfill = {
                "mode": "backfill",
                "start_date": str(start_date),
                "end_date": str(end_date),
                "start_time": str(start_time),
                "end_time": str(end_time),
                "crashes_columns": selected_crashes_bf,
                "vehicles_columns": selected_vehicles_bf if include_vehicles_bf else [],
                "people_columns": selected_people_bf if include_people_bf else [],
                "include_vehicles": include_vehicles_bf,
                "include_people": include_people_bf
            }
            st.json(preview_json_backfill)

        # Buttons
        button_col1_bf, button_col2_bf = st.columns(2)
        with button_col1_bf:
            if st.button("📤 Publish to RabbitMQ", key="publish_backfill"):
                with st.spinner("Publishing job..."):
                    result = call_api("/api/fetch/publish", method="POST", data=preview_json_backfill)
                    if result and result.get("status") == "success":
                        st.success(f"✅ Job queued! CorrID: **{result['corrid']}**")
                        st.json(result)
                    else:
                        st.error("Failed to publish job.")

        with button_col2_bf:
            if st.button("🔄 Reset form", key="reset_backfill"):
                st.rerun()

# =============================
# Tab 4: Scheduler
# =============================
def render_scheduler_tab():
    st.markdown('<div class="sub-header">⏰ Scheduler</div>', unsafe_allow_html=True)

    # Refresh button
    _col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_scheduler"):
            st.rerun()

    # Frequency selector
    st.markdown("### ➕ Create New Schedule")
    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        frequency = st.selectbox("Select Frequency", ["Daily", "Weekly", "Custom cron"], key="sched_freq")

    with freq_col2:
        time_picker = st.time_input("Run time", value=datetime.strptime("09:00:00", "%H:%M:%S").time(), key="sched_time")

    # Custom cron (if selected)
    if frequency == "Custom cron":
        cron_string = st.text_input("Cron expression", value="0 9 * * *", key="cron_expr")
    else:
        if frequency == "Daily":
            cron_string = f"{time_picker.minute} {time_picker.hour} * * *"
        else:  # Weekly
            cron_string = f"{time_picker.minute} {time_picker.hour} * * 1"  # Every Monday

    st.code(f"Cron: {cron_string}", language="text")

    # Config type
    config_type = st.selectbox("Config Type", ["streaming", "backfill"], key="sched_config")

    # Load job config based on type
    if config_type == "streaming":
        since_days = st.number_input("Since days", min_value=1, max_value=3650, value=30, key="sched_since_days")
        job_config = {
            "mode": "streaming",
            "since_days": since_days,
            "crashes_columns": DEFAULT_CRASHES_COLS,
            "vehicles_columns": DEFAULT_VEHICLES_COLS if st.checkbox("Include vehicles", value=True, key="sched_vehicles") else [],
            "people_columns": DEFAULT_PEOPLE_COLS if st.checkbox("Include people", value=True, key="sched_people") else [],
            "include_vehicles": True,
            "include_people": True
        }
    else:  # backfill
        st.warning("Backfill schedules require careful date range selection")
        job_config = {
            "mode": "backfill",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "start_time": "00:00:00",
            "end_time": "23:59:59",
            "crashes_columns": DEFAULT_CRASHES_COLS,
            "vehicles_columns": [],
            "people_columns": [],
            "include_vehicles": False,
            "include_people": False
        }

    if st.button("➕ Create Schedule", key="create_sched"):
        with st.spinner("Creating schedule..."):
            result = call_api("/api/schedule/create", method="POST", data={
                "cron_expr": cron_string,
                "config_type": config_type,
                "job_config": job_config
            })
            if result and result.get("status") == "success":
                st.success(f"✅ Schedule created: {frequency} at {time_picker} ({config_type})")
                st.json(result.get("schedule"))
                st.rerun()
            else:
                st.error("Failed to create schedule")

    st.markdown("---")

    # Active Schedules Table
    st.markdown("### 📋 Active Schedules")

    schedules_data = call_api("/api/schedule/list")

    if schedules_data and schedules_data.get("schedules"):
        schedules = schedules_data["schedules"]

        for schedule in schedules:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 2, 1, 1])

                with col1:
                    st.text(f"ID: {schedule['id']}")
                with col2:
                    st.text(f"Cron: {schedule['cron_expr']}")
                with col3:
                    st.text(f"Type: {schedule['config_type']}")
                with col4:
                    last_run = schedule.get("last_run", "Never")
                    if last_run and last_run != "Never":
                        last_run = last_run[:19].replace("T", " ")
                    st.text(f"Last: {last_run}")
                with col5:
                    status = "✅ Active" if schedule.get("enabled") else "⏸️ Paused"
                    st.text(status)
                with col6:
                    if st.button("🗑️", key=f"delete_{schedule['id']}", help="Delete schedule"):
                        with st.spinner("Deleting..."):
                            result = call_api(f"/api/schedule/{schedule['id']}", method="DELETE")
                            if result:
                                st.success("Deleted!")
                                st.rerun()

                st.markdown("---")

        st.caption(f"Total schedules: {len(schedules)}")
    else:
        st.info("No schedules created yet. Create your first schedule above.")

# =============================
# Tab 5: EDA
# =============================
def render_eda_tab():
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_eda"):
            st.cache_data.clear()
            st.rerun()

    # Check if Gold DB exists
    gold_status = get_gold_status()
    if not gold_status or not gold_status.get("exists"):
        st.warning("Gold database does not exist. Please fetch and process some data first.")
        return

    # Row limit selector
    st.markdown("### 📊 Data Loading")
    row_limit = st.select_slider(
        "Number of rows to load for analysis",
        options=[1000, 5000, 10000, 20000, 50000, 100000],
        value=10000,
        key="eda_row_limit"
    )
    st.caption(f"Loading {row_limit:,} most recent crashes. Larger samples = more accurate patterns but slower loading.")

    # Load data
    with st.spinner(f"Loading {row_limit:,} crashes from Gold database..."):
        query = f'SELECT * FROM "gold"."main"."crashes" ORDER BY crash_date DESC LIMIT {row_limit}'
        data_result = get_gold_data(query)

    if not data_result or not data_result.get("data"):
        st.error("No data available in Gold database.")
        return

    df = pd.DataFrame(data_result["data"])

    # Get total count from database
    total_query = 'SELECT COUNT(*) as total FROM "gold"."main"."crashes"'
    total_result = get_gold_data(total_query)
    total_in_db = total_result["data"][0]["total"] if total_result and total_result.get("data") else len(df)

    st.success(f"Loaded {len(df):,} crashes from Gold database (Total in DB: {total_in_db:,})")

    # Summary Statistics
    st.markdown("### 📈 Summary Statistics")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Total Crashes (in sample)", f"{len(df):,}")
        st.caption(f"Total in DB: {total_in_db:,}")
    with summary_col2:
        if "injury" in df.columns:
            serious_count = df["injury"].sum() if df["injury"].notna().any() else 0
            st.metric("Injuries", int(serious_count))
    with summary_col3:
        if "injury" in df.columns:
            serious_rate = (df["injury"].mean() * 100) if df["injury"].notna().any() else 0
            st.metric("Injury Rate", f"{serious_rate:.1f}%")
    with summary_col4:
        if "crash_date" in df.columns:
            date_range = f"{df['crash_date'].min()[:10]} to {df['crash_date'].max()[:10]}"
            st.metric("Date Range", "")
            st.caption(date_range)

    st.markdown("---")

    # Visualizations
    st.markdown("### 📊 Visualizations")

    # Filter data for injury if available
    if "injury" in df.columns:
        df["injury_label"] = df["injury"].map({0: "Non-injury", 1: "Injury"})

    # Visualization 1: Histogram - Posted Speed Limit
    if "posted_speed_limit" in df.columns:
        st.markdown("#### 1️⃣ Distribution of Posted Speed Limit")
        fig1 = px.histogram(
            df,
            x="posted_speed_limit",
            color="injury_label" if "injury_label" in df.columns else None,
            nbins=20,
            title="Posted Speed Limit Distribution by Injury",
            labels={"posted_speed_limit": "Speed Limit (mph)", "count": "Number of Crashes"},
            barmode="overlay",
            opacity=0.7
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Calculate data-driven insight
        if "injury_label" in df.columns:
            speed_analysis = df.groupby("posted_speed_limit")["injury"].agg(['sum', 'count', 'mean']).reset_index()
            speed_analysis['rate'] = speed_analysis['mean'] * 100
            low_speed = speed_analysis[speed_analysis['posted_speed_limit'] <= 30]['rate'].mean()
            high_speed = speed_analysis[speed_analysis['posted_speed_limit'] >= 45]['rate'].mean()
            st.caption(f"💡 Insight: Serious injury rate at ≤30 mph: {low_speed:.1f}% vs. ≥45 mph: {high_speed:.1f}%. Higher speeds show {high_speed/low_speed:.1f}x increased injury risk.")
        else:
            st.caption("💡 Insight: Speed limit distribution varies across crash locations.")

    # Visualization 2: Bar Chart - Weather Condition
    if "weather_condition" in df.columns:
        st.markdown("#### 2️⃣ Crashes by Weather Condition")
        weather_counts = df.groupby(["weather_condition", "injury_label"]).size().reset_index(name="count") if "injury_label" in df.columns else df["weather_condition"].value_counts().reset_index()
        fig2 = px.bar(
            weather_counts.head(20),
            x="weather_condition" if "index" not in weather_counts.columns else "index",
            y="count",
            color="injury_label" if "injury_label" in weather_counts.columns else None,
            title="Crash Counts by Weather Condition",
            labels={"weather_condition": "Weather", "count": "Crashes"},
            barmode="group"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Calculate data-driven insight
        if "injury_label" in df.columns:
            weather_analysis = df.groupby("weather_condition")["injury"].agg(['sum', 'count', 'mean']).reset_index()
            weather_analysis['rate'] = weather_analysis['mean'] * 100
            weather_analysis = weather_analysis.sort_values('count', ascending=False).head(5)
            top_weather = weather_analysis.iloc[0]['weather_condition']
            top_rate = weather_analysis.iloc[0]['rate']
            highest_rate_weather = weather_analysis.sort_values('rate', ascending=False).iloc[0]
            st.caption(f"💡 Insight: '{top_weather}' has most crashes ({weather_analysis.iloc[0]['count']:.0f}). Highest injury rate: '{highest_rate_weather['weather_condition']}' at {highest_rate_weather['rate']:.1f}%.")
        else:
            st.caption("💡 Insight: Weather conditions vary across crash incidents.")

    # Visualization 3: Line Chart - Crash Hour
    if "hour" in df.columns:
        st.markdown("#### 3️⃣ Crash Pattern by Hour of Day")
        hour_data = df.groupby("hour").size().reset_index(name="count")
        fig3 = px.line(
            hour_data,
            x="hour",
            y="count",
            title="Crash Frequency by Hour of Day",
            labels={"hour": "Hour", "count": "Number of Crashes"},
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Calculate data-driven insight
        peak_hour = hour_data.loc[hour_data['count'].idxmax()]
        low_hour = hour_data.loc[hour_data['count'].idxmin()]
        st.caption(f"💡 Insight: Peak crash hour is {int(peak_hour['hour'])}:00 with {int(peak_hour['count'])} crashes. Lowest is {int(low_hour['hour'])}:00 with {int(low_hour['count'])} crashes ({peak_hour['count']/low_hour['count']:.1f}x difference).")

    # Visualization 4: Pie Chart - Injury Distribution
    if "injury_label" in df.columns:
        st.markdown("#### 4️⃣ Injury Class Distribution")
        serious_counts = df["injury_label"].value_counts().reset_index()
        serious_counts.columns = ["Outcome", "Count"]
        fig4 = px.pie(
            serious_counts,
            names="Outcome",
            values="Count",
            title="Proportion of Injury vs No Injury",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Calculate data-driven insight
        serious_pct = (df["injury"].sum() / len(df)) * 100
        imbalance_ratio = (len(df) - df["injury"].sum()) / df["injury"].sum()
        st.caption(f"💡 Insight: Injuries represent {serious_pct:.1f}% of crashes (imbalance ratio ~1:{imbalance_ratio:.0f}). Class balancing techniques recommended for ML models.")

    # Visualization 5: Heatmap - Hour × Day of Week
    if "hour" in df.columns and "day_of_week" in df.columns:
        st.markdown("#### 5️⃣ Crash Heatmap: Hour × Day of Week")
        heatmap_data = df.groupby(["hour", "day_of_week"]).size().reset_index(name="count")
        heatmap_pivot = heatmap_data.pivot(index="hour", columns="day_of_week", values="count").fillna(0)
        fig5 = px.imshow(
            heatmap_pivot,
            title="Crash Frequency Heatmap (Hour × Day of Week)",
            labels={"x": "Day of Week", "y": "Hour", "color": "Crashes"},
            color_continuous_scale="YlOrRd"
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Calculate data-driven insight
        peak_cell = heatmap_data.loc[heatmap_data['count'].idxmax()]
        st.caption(f"💡 Insight: Highest crash density at day={int(peak_cell['day_of_week'])}, hour={int(peak_cell['hour'])}:00 with {int(peak_cell['count'])} crashes. Weekday PM commute hours dominate.")

    # Visualization 6: Box Plot - Age Distribution by Injury
    if "ppl_age_mean" in df.columns and "injury_label" in df.columns:
        st.markdown("#### 6️⃣ Age Distribution by Injury Outcome")
        fig6 = px.box(
            df.dropna(subset=["ppl_age_mean"]),
            x="injury_label",
            y="ppl_age_mean",
            title="People Age Distribution by Injury",
            labels={"injury_label": "Outcome", "ppl_age_mean": "Average Age"},
            color="injury_label"
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Calculate data-driven insight
        age_df = df.dropna(subset=["ppl_age_mean"])
        serious_age = age_df[age_df["injury"] == 1]["ppl_age_mean"].median()
        nonserious_age = age_df[age_df["injury"] == 0]["ppl_age_mean"].median()
        st.caption(f"💡 Insight: Median age in injury crashes: {serious_age:.1f} years vs. no injury: {nonserious_age:.1f} years. Age difference: {abs(serious_age - nonserious_age):.1f} years.")

    # Visualization 7: Bar Chart - First Crash Type
    if "first_crash_type" in df.columns:
        st.markdown("#### 7️⃣ Top Crash Types")
        crash_type_counts = df["first_crash_type"].value_counts().head(10).reset_index()
        crash_type_counts.columns = ["Crash Type", "Count"]
        fig7 = px.bar(
            crash_type_counts,
            x="Count",
            y="Crash Type",
            orientation="h",
            title="Top 10 Crash Types",
            color="Count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig7, use_container_width=True)

        # Calculate data-driven insight
        top_type = crash_type_counts.iloc[0]
        second_type = crash_type_counts.iloc[1]
        pct_of_total = (top_type['Count'] / len(df)) * 100
        st.caption(f"💡 Insight: '{top_type['Crash Type']}' is most common ({int(top_type['Count'])} crashes, {pct_of_total:.1f}% of total). Second: '{second_type['Crash Type']}' ({int(second_type['Count'])}).")

    # Visualization 8: Scatter - Latitude × Longitude
    if "latitude" in df.columns and "longitude" in df.columns:
        st.markdown("#### 8️⃣ Geographic Crash Distribution")
        sample_size = min(1000, len(df))
        map_sample = df.sample(sample_size)
        fig8 = px.scatter_mapbox(
            map_sample,
            lat="latitude",
            lon="longitude",
            color="injury_label" if "injury_label" in df.columns else None,
            title="Crash Locations (Sampled)",
            mapbox_style="carto-positron",
            zoom=10,
            height=500,
            color_discrete_map={"Non-injury": "blue", "Injury": "red"} if "injury_label" in df.columns else None
        )
        st.plotly_chart(fig8, use_container_width=True)

        # Calculate data-driven insight
        if "injury_label" in df.columns:
            serious_sampled = map_sample["injury"].sum()
            serious_pct_map = (serious_sampled / len(map_sample)) * 100
            st.caption(f"💡 Insight: Sampled {sample_size} crashes. {int(serious_sampled)} injuries shown in red ({serious_pct_map:.1f}%). Hotspots visible along major arterials and downtown.")
        else:
            st.caption(f"💡 Insight: Sampled {sample_size} crashes across Chicago. Geographic patterns reveal high-traffic corridors.")

    # Visualization 9: Bar Chart - Lighting Condition
    if "lighting_condition" in df.columns:
        st.markdown("#### 9️⃣ Crashes by Lighting Condition")
        lighting_counts = df["lighting_condition"].value_counts().head(8).reset_index()
        lighting_counts.columns = ["Lighting", "Count"]
        fig9 = px.bar(
            lighting_counts,
            x="Lighting",
            y="Count",
            title="Crash Counts by Lighting Condition",
            color="Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig9, use_container_width=True)

        # Calculate data-driven insight
        if "injury_label" in df.columns:
            lighting_analysis = df.groupby("lighting_condition")["injury"].agg(['sum', 'count', 'mean']).reset_index()
            lighting_analysis['rate'] = lighting_analysis['mean'] * 100
            lighting_analysis = lighting_analysis.sort_values('count', ascending=False)
            top_light = lighting_analysis.iloc[0]
            worst_light = lighting_analysis.sort_values('rate', ascending=False).iloc[0]
            st.caption(f"💡 Insight: '{top_light['lighting_condition']}' has most crashes ({int(top_light['count'])}). Highest injury rate: '{worst_light['lighting_condition']}' at {worst_light['rate']:.1f}%.")
        else:
            st.caption(f"💡 Insight: '{lighting_counts.iloc[0]['Lighting']}' accounts for {int(lighting_counts.iloc[0]['Count'])} crashes.")

    # Visualization 10: Histogram - Vehicle Count
    if "veh_count" in df.columns:
        st.markdown("#### 🔟 Distribution of Vehicle Count per Crash")
        fig10 = px.histogram(
            df,
            x="veh_count",
            title="Number of Vehicles Involved per Crash",
            labels={"veh_count": "Vehicle Count", "count": "Frequency"},
            nbins=10
        )
        st.plotly_chart(fig10, use_container_width=True)

        # Calculate data-driven insight
        mode_veh = df["veh_count"].mode()[0]
        mode_count = (df["veh_count"] == mode_veh).sum()
        pct_mode = (mode_count / len(df)) * 100
        st.caption(f"💡 Insight: {int(mode_veh)}-vehicle crashes are most common ({int(mode_count)} incidents, {pct_mode:.1f}% of total). Single-vehicle: {(df['veh_count'] == 1).sum()} crashes.")

    # Visualization 11: Donut Chart - Weekend vs Weekday
    if "is_weekend" in df.columns:
        st.markdown("#### 1️⃣1️⃣ Weekend vs Weekday Crashes")
        weekend_counts = df["is_weekend"].value_counts().reset_index()
        weekend_counts.columns = ["Is Weekend", "Count"]
        weekend_counts["Is Weekend"] = weekend_counts["Is Weekend"].map({0: "Weekday", 1: "Weekend"})
        fig11 = px.pie(
            weekend_counts,
            names="Is Weekend",
            values="Count",
            title="Crashes: Weekend vs Weekday",
            hole=0.4
        )
        st.plotly_chart(fig11, use_container_width=True)

        # Calculate data-driven insight
        weekday_crashes = (df["is_weekend"] == 0).sum()
        weekend_crashes = (df["is_weekend"] == 1).sum()
        weekday_pct = (weekday_crashes / len(df)) * 100
        st.caption(f"💡 Insight: Weekdays: {int(weekday_crashes)} crashes ({weekday_pct:.1f}%). Weekends: {int(weekend_crashes)} crashes ({100-weekday_pct:.1f}%). Weekdays have {weekday_crashes/weekend_crashes:.1f}x more crashes.")

    # Visualization 12: Bar Chart - Hour Bin
    if "hour_bin" in df.columns:
        st.markdown("#### 1️⃣2️⃣ Crashes by Time of Day (Hour Bin)")
        hour_bin_counts = df["hour_bin"].value_counts().reset_index()
        hour_bin_counts.columns = ["Hour Bin", "Count"]
        fig12 = px.bar(
            hour_bin_counts,
            x="Hour Bin",
            y="Count",
            title="Crash Counts by Time of Day",
            color="Count"
        )
        st.plotly_chart(fig12, use_container_width=True)

        # Calculate data-driven insight
        top_bin = hour_bin_counts.iloc[0]
        low_bin = hour_bin_counts.iloc[-1]
        st.caption(f"💡 Insight: '{top_bin['Hour Bin']}' has most crashes ({int(top_bin['Count'])}). Quietest: '{low_bin['Hour Bin']}' ({int(low_bin['Count'])} crashes, {top_bin['Count']/low_bin['Count']:.1f}x difference).")

    # Visualization 13: Line Chart - Injury Rate by Hour
    if "hour" in df.columns and "injury" in df.columns:
        st.markdown("#### 1️⃣3️⃣ Injury Rate by Hour")
        hour_serious = df.groupby("hour").agg({"injury": ["sum", "count"]}).reset_index()
        hour_serious.columns = ["hour", "serious_count", "total_count"]
        hour_serious["rate"] = (hour_serious["serious_count"] / hour_serious["total_count"] * 100).fillna(0)
        fig13 = px.line(
            hour_serious,
            x="hour",
            y="rate",
            title="Injury Rate by Hour of Day",
            labels={"hour": "Hour", "rate": "Injury Rate (%)"},
            markers=True
        )
        st.plotly_chart(fig13, use_container_width=True)

        # Calculate data-driven insight
        max_rate_hour = hour_serious.loc[hour_serious["rate"].idxmax()]
        min_rate_hour = hour_serious.loc[hour_serious["rate"].idxmin()]
        st.caption(f"💡 Insight: Peak injury rate at {int(max_rate_hour['hour'])}:00 ({max_rate_hour['rate']:.1f}%). Lowest at {int(min_rate_hour['hour'])}:00 ({min_rate_hour['rate']:.1f}%). Late-night/early-morning hours show elevated risk.")

    # Visualization 14: Stacked Bar - Crash Type by Injury
    if "first_crash_type" in df.columns and "injury_label" in df.columns:
        st.markdown("#### 1️⃣4️⃣ Crash Type Distribution by Injury")
        crash_type_serious = df.groupby(["first_crash_type", "injury_label"]).size().reset_index(name="count")
        top_crash_types = df["first_crash_type"].value_counts().head(8).index
        crash_type_serious = crash_type_serious[crash_type_serious["first_crash_type"].isin(top_crash_types)]
        fig14 = px.bar(
            crash_type_serious,
            x="first_crash_type",
            y="count",
            color="injury_label",
            title="Top Crash Types by Injury Outcome",
            labels={"first_crash_type": "Crash Type", "count": "Count"},
            barmode="stack"
        )
        st.plotly_chart(fig14, use_container_width=True)

        # Calculate data-driven insight
        crash_type_rates = df.groupby("first_crash_type")["injury"].agg(['sum', 'count', 'mean']).reset_index()
        crash_type_rates['rate'] = crash_type_rates['mean'] * 100
        crash_type_rates = crash_type_rates[crash_type_rates['first_crash_type'].isin(top_crash_types)]
        deadliest_type = crash_type_rates.sort_values('rate', ascending=False).iloc[0]
        st.caption(f"💡 Insight: Among top crash types, '{deadliest_type['first_crash_type']}' has highest injury rate at {deadliest_type['rate']:.1f}% ({int(deadliest_type['sum'])} of {int(deadliest_type['count'])} crashes).")

    # Visualization 15: Violin Plot - Speed Limit by Injury
    if "posted_speed_limit" in df.columns and "injury_label" in df.columns:
        st.markdown("#### 1️⃣5️⃣ Speed Limit Distribution by Injury (Violin Plot)")
        fig15 = px.violin(
            df.dropna(subset=["posted_speed_limit"]),
            x="injury_label",
            y="posted_speed_limit",
            title="Posted Speed Limit Distribution by Injury",
            labels={"injury_label": "Outcome", "posted_speed_limit": "Speed Limit (mph)"},
            color="injury_label",
            box=True
        )
        st.plotly_chart(fig15, use_container_width=True)

        # Calculate data-driven insight
        speed_df = df.dropna(subset=["posted_speed_limit"])
        serious_median_speed = speed_df[speed_df["injury"] == 1]["posted_speed_limit"].median()
        nonserious_median_speed = speed_df[speed_df["injury"] == 0]["posted_speed_limit"].median()
        serious_mean_speed = speed_df[speed_df["injury"] == 1]["posted_speed_limit"].mean()
        nonserious_mean_speed = speed_df[speed_df["injury"] == 0]["posted_speed_limit"].mean()

        # Count distribution at high speeds
        high_speed_serious = speed_df[(speed_df["injury"] == 1) & (speed_df["posted_speed_limit"] >= 45)].shape[0]
        high_speed_nonserious = speed_df[(speed_df["injury"] == 0) & (speed_df["posted_speed_limit"] >= 45)].shape[0]

        st.caption(f"💡 Insight: Both injury and no injury crashes show similar median speed limits ({serious_median_speed:.0f} mph vs {nonserious_median_speed:.0f} mph), but no injury crashes occur more frequently at higher speeds (≥45 mph: {high_speed_nonserious} no injury vs {high_speed_serious} injury). Volume differences, not speed alone, drive the distribution.")

# =============================
# Tab 6: Model
# =============================
def render_model_tab():
    """Render the Model tab for ML model predictions and evaluation."""
    st.markdown('<div class="sub-header">🤖 ML Model</div>', unsafe_allow_html=True)

    # ========== Section 1: Model Summary ==========
    st.markdown("### 📋 Model Summary")

    # Load model using cached helper
    model = load_model()

    # Expected features (from ML notebook)
    NUMERICAL_FEATURES = [
        'posted_speed_limit', 'num_units', 'crash_hour', 'crash_day_of_week',
        'crash_month', 'beat_of_occurrence', 'intersection_related_i', 'hit_and_run_i',
        'latitude', 'longitude', 'lat_bin', 'lng_bin', 'veh_count', 'ppl_count',
        'year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend',
        'veh_truck_i', 'veh_mc_i', 'ppl_age_mean', 'ppl_age_min', 'ppl_age_max'
    ]

    CATEGORICAL_FEATURES = [
        'traffic_control_device', 'device_condition', 'weather_condition',
        'lighting_condition', 'first_crash_type', 'trafficway_type',
        'alignment', 'roadway_surface_cond', 'road_defect', 'hour_bin'
    ]

    DECISION_THRESHOLD = 0.3449

    # Display model info
    model_type = type(model).__name__
    base_estimator_type = type(model.estimator).__name__ if hasattr(model, 'estimator') else "Unknown"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", model_type)
    with col2:
        st.metric("Base Estimator", base_estimator_type)
    with col3:
        st.metric("Decision Threshold", f"{DECISION_THRESHOLD:.4f}")

    # Feature info
    st.markdown("#### Expected Features")
    st.info(f"""
    **Total Features**: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)} features
    - **Numerical**: {len(NUMERICAL_FEATURES)} features
    - **Categorical**: {len(CATEGORICAL_FEATURES)} features

    **Target**: `injury` - Binary classification (0 = No Injury, 1 = Injury)

    **Important**: The model expects **raw column values**. One-hot encoding, imputation, and scaling are handled **inside the pipeline** automatically.
    """)

    with st.expander("View Feature List"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Numerical Features (25)**")
            for feat in NUMERICAL_FEATURES:
                st.text(f"• {feat}")
        with col_b:
            st.markdown("**Categorical Features (10)**")
            for feat in CATEGORICAL_FEATURES:
                st.text(f"• {feat}")

    st.markdown("---")

    # ========== Section 2: Data Selection ==========
    st.markdown("### 📂 Data Selection & Prediction")

    # Mode selection
    data_mode = st.radio(
        "Select Data Source",
        options=["Gold Database", "Test CSV Upload"],
        horizontal=True,
        help="Choose between querying Gold DB directly or uploading a test CSV file"
    )

    if data_mode == "Gold Database":
        st.markdown("#### Configure Gold Database Query")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )

        max_rows = st.number_input(
            "Maximum Rows to Score",
            min_value=100,
            max_value=50000,
            value=5000,
            step=1000,
            help="Limit the number of rows to score (for performance)"
        )

        st.info("💡 **Efficient Processing**: The API will query the Gold database directly and return only predictions (no data transfer over HTTP).")

    else:  # Test CSV Upload
        st.markdown("#### Upload Test CSV File")

        st.info("""
        **Requirements for test CSV:**
        - Must be a `.csv` file
        - Must contain all 35 required features (see Model Summary above)
        - Must include the `injury` column (0 or 1) for metrics calculation
        - Use `export_test_data.py` script to generate a valid test file from Gold DB
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with features and injury label"
        )

        if uploaded_file is not None:
            # Validate file extension
            if not uploaded_file.name.endswith('.csv'):
                st.error("❌ **Invalid file type!** Only CSV (`.csv`) files are allowed.")
                st.stop()

            try:
                # Load CSV
                test_df = pd.read_csv(uploaded_file)

                # Store in session state
                st.session_state['test_csv_data'] = test_df

                st.success(f"✅ Loaded {len(test_df)} rows from `{uploaded_file.name}`")

                # Validate features
                required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                missing_features = [f for f in required_features if f not in test_df.columns]

                if missing_features:
                    st.error(f"❌ **Missing features**: {', '.join(missing_features[:10])}")
                    st.warning("The uploaded CSV is missing required features. Please ensure all 35 features are present.")
                else:
                    st.success("✅ All required features present")

                # Check for injury column
                if 'injury' not in test_df.columns:
                    st.warning("⚠️ No `injury` column found. Metrics will not be computed.")
                else:
                    # Show label distribution
                    injury_counts = test_df['injury'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No Injury (0)", injury_counts.get(0, 0))
                    with col2:
                        st.metric("Injury (1)", injury_counts.get(1, 0))

                # Show preview
                with st.expander("📊 Data Preview (first 10 rows)"):
                    st.dataframe(test_df.head(10))

            except Exception as e:
                st.error(f"❌ Failed to load CSV: {e}")
                st.stop()

    st.markdown("---")

    # ========== Section 3: Prediction & Metrics ==========
    st.markdown("### 🎯 Prediction & Metrics")

    # Static test metrics (from notebook)
    st.markdown("#### 📊 Static Metrics (Test Set Performance)")
    st.caption("These metrics were computed on the held-out test set during model training.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PR-AUC", "0.5846")
    with col2:
        st.metric("F1 Score", "0.530")
    with col3:
        st.metric("ROC-AUC", "0.847")
    with col4:
        st.metric("Accuracy", "0.86")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", "0.581")
    with col2:
        st.metric("Recall", "0.487")
    with col3:
        st.metric("Threshold", "0.3449")

    # Confusion matrix visualization
    with st.expander("📈 Test Set Confusion Matrix"):
        st.markdown("""
        **Confusion Matrix (Test Set, n=83,846)**

        |                | Predicted: No Injury | Predicted: Injury |
        |----------------|---------------------|-------------------|
        | **Actual: No Injury** | 65,669 (TN)        | 4,943 (FP)        |
        | **Actual: Injury**    | 6,750 (FN)         | 6,484 (TP)        |

        **Key Insight**: The model correctly identifies 6,484 injury crashes but misses 6,750 (51.3% of actual injuries).
        False alarms: 4,943 cases flagged as injury that were not.
        """)

    st.markdown("---")

    # Run predictions button
    predict_enabled = (data_mode == "Gold Database") or (data_mode == "Test CSV Upload" and 'test_csv_data' in st.session_state)

    if not predict_enabled and data_mode == "Test CSV Upload":
        st.info("👆 Upload a CSV file above to enable predictions")

    if st.button("🚀 Run Predictions", type="primary", disabled=not predict_enabled):
        if data_mode == "Gold Database":
            # Mode 1: Query Gold DB directly using cached connection
            with st.spinner("Querying Gold DB and running predictions..."):
                try:
                    import numpy as np
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                    # Build query
                    required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                    feature_str = ", ".join([f'"{f}"' for f in required_features])
                    query = f"""
                    SELECT {feature_str}, "injury"
                    FROM "gold"."main"."crashes"
                    WHERE crash_date BETWEEN '{start_date}' AND '{end_date}'
                    LIMIT {max_rows}
                    """

                    # Query Gold DB
                    df = query_gold_db(query)

                    if df is None or df.empty:
                        st.warning("No data found for the selected date range.")
                        st.stop()

                    # Check for labels
                    has_labels = 'injury' in df.columns
                    y_true = df['injury'].values if has_labels else None

                    # Prepare features
                    X = df[required_features].copy()
                    X = X.replace([np.inf, -np.inf], np.nan)

                    # Get predictions
                    import time
                    pred_start = time.time()
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    proba_positive = probabilities[:, 1]
                    predictions_thresholded = (proba_positive >= DECISION_THRESHOLD).astype(int)
                    prediction_latency.observe(time.time() - pred_start)

                    # Calculate metrics if labels exist
                    metrics = None
                    if has_labels and y_true is not None:
                        valid_idx = ~np.isnan(y_true)
                        if valid_idx.sum() > 0:
                            y_true_valid = y_true[valid_idx].astype(int)
                            y_pred_valid = predictions_thresholded[valid_idx]

                            metrics = {
                                "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
                                "precision": float(precision_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "recall": float(recall_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "f1_score": float(f1_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid).tolist()
                            }

                            # Update Prometheus metrics
                            model_accuracy.set(metrics["accuracy"])
                            model_precision.set(metrics["precision"])
                            model_recall.set(metrics["recall"])
                            predictions_made.inc(len(predictions_thresholded))

                    # Prediction distribution
                    unique, counts = np.unique(predictions_thresholded, return_counts=True)
                    pred_distribution = {"0": 0, "1": 0}
                    for k, v in zip(unique, counts):
                        pred_distribution[str(int(k))] = int(v)

                    # Probability statistics
                    prob_hist, bin_edges = np.histogram(proba_positive, bins=50, range=(0, 1))

                    # Store results
                    pred_result = {
                        "success": True,
                        "row_count": len(df),
                        "predictions": predictions_thresholded.tolist(),
                        "probabilities": proba_positive.tolist(),
                        "prediction_distribution": pred_distribution,
                        "probability_stats": {
                            "min": float(np.min(proba_positive)),
                            "max": float(np.max(proba_positive)),
                            "mean": float(np.mean(proba_positive)),
                            "median": float(np.median(proba_positive)),
                            "std": float(np.std(proba_positive))
                        },
                        "probability_histogram": {
                            "counts": prob_hist.tolist(),
                            "bin_edges": bin_edges.tolist()
                        },
                        "threshold": DECISION_THRESHOLD,
                        "metrics": metrics,
                        "has_labels": has_labels
                    }

                    st.session_state['pred_result'] = pred_result
                    st.session_state['data_source'] = 'gold_db'
                    st.success(f"✅ Predictions complete for {len(df)} rows")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

        else:  # Test CSV Upload mode
            # Mode 2: Use uploaded CSV data with direct model loading
            with st.spinner("Running predictions on uploaded CSV..."):
                try:
                    import numpy as np
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                    test_df = st.session_state['test_csv_data']
                    required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

                    # Prepare features
                    X = test_df[required_features].copy()
                    X = X.replace([np.inf, -np.inf], np.nan)

                    # Get predictions
                    import time
                    pred_start = time.time()
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    proba_positive = probabilities[:, 1]
                    predictions_thresholded = (proba_positive >= DECISION_THRESHOLD).astype(int)
                    prediction_latency.observe(time.time() - pred_start)

                    # Calculate metrics if injury column exists
                    metrics = None
                    has_labels = 'injury' in test_df.columns

                    if has_labels:
                        y_true = test_df['injury'].values
                        valid_idx = ~np.isnan(y_true)
                        if valid_idx.sum() > 0:
                            y_true_valid = y_true[valid_idx].astype(int)
                            y_pred_valid = predictions_thresholded[valid_idx]

                            metrics = {
                                "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
                                "precision": float(precision_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "recall": float(recall_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "f1_score": float(f1_score(y_true_valid, y_pred_valid, zero_division=0)),
                                "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid).tolist()
                            }

                            # Update Prometheus metrics
                            model_accuracy.set(metrics["accuracy"])
                            model_precision.set(metrics["precision"])
                            model_recall.set(metrics["recall"])
                            predictions_made.inc(len(predictions_thresholded))

                    # Build prediction distribution
                    unique, counts = np.unique(predictions_thresholded, return_counts=True)
                    pred_distribution = {"0": 0, "1": 0}
                    for k, v in zip(unique, counts):
                        pred_distribution[str(int(k))] = int(v)

                    # Probability statistics
                    prob_hist, bin_edges = np.histogram(proba_positive, bins=50, range=(0, 1))

                    csv_result = {
                        "success": True,
                        "row_count": len(test_df),
                        "predictions": predictions_thresholded.tolist(),
                        "probabilities": proba_positive.tolist(),
                        "prediction_distribution": pred_distribution,
                        "probability_stats": {
                            "min": float(np.min(proba_positive)),
                            "max": float(np.max(proba_positive)),
                            "mean": float(np.mean(proba_positive)),
                            "median": float(np.median(proba_positive)),
                            "std": float(np.std(proba_positive))
                        },
                        "probability_histogram": {
                            "counts": prob_hist.tolist(),
                            "bin_edges": bin_edges.tolist()
                        },
                        "threshold": DECISION_THRESHOLD,
                        "metrics": metrics,
                        "has_labels": has_labels
                    }

                    st.session_state['pred_result'] = csv_result
                    st.session_state['data_source'] = 'csv_upload'
                    st.success(f"✅ Predictions complete for {len(test_df)} rows")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()

    # Display predictions
    if 'pred_result' in st.session_state:
        pred_result = st.session_state['pred_result']

        st.markdown("---")
        st.markdown("#### 🔮 Live Predictions")

        # Prediction distribution (JSON converts int keys to strings)
        pred_dist = pred_result.get('prediction_distribution', {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted: No Injury (0)", pred_dist.get("0", pred_dist.get(0, 0)))
        with col2:
            st.metric("Predicted: Injury (1)", pred_dist.get("1", pred_dist.get(1, 0)))

        # Probability statistics
        prob_stats = pred_result.get('probability_stats', {})
        st.markdown("#### 📊 Probability Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{prob_stats.get('min', 0):.3f}")
        with col2:
            st.metric("Max", f"{prob_stats.get('max', 0):.3f}")
        with col3:
            st.metric("Mean", f"{prob_stats.get('mean', 0):.3f}")
        with col4:
            st.metric("Median", f"{prob_stats.get('median', 0):.3f}")
        with col5:
            st.metric("Std Dev", f"{prob_stats.get('std', 0):.3f}")

        # Probability distribution histogram
        st.markdown("#### 📊 Probability Distribution")
        prob_hist = pred_result.get('probability_histogram', {})
        if prob_hist:
            bin_edges = prob_hist.get('bin_edges', [])
            counts = prob_hist.get('counts', [])

            # Create midpoints for plotting
            bin_midpoints = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

            fig = go.Figure(data=[go.Bar(x=bin_midpoints, y=counts, width=0.018)])
            fig.add_vline(x=0.3449, line_dash="dash", line_color="red",
                         annotation_text="Decision Threshold (0.3449)", annotation_position="top right")
            fig.update_layout(
                title="Distribution of Predicted Probabilities",
                xaxis_title="Predicted Probability (Injury)",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Live metrics (if labels exist)
        if pred_result.get('has_labels', False) and pred_result.get('metrics'):
            metrics = pred_result['metrics']

            st.markdown("---")
            st.markdown("#### 📈 Live Metrics (on Gold data)")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")

            # Confusion matrix
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            st.markdown("**Confusion Matrix (Live Data)**")
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: No Injury', 'Actual: Injury'],
                columns=['Predicted: No Injury', 'Predicted: Injury']
            )
            st.dataframe(cm_df)

            # Comparison table
            st.markdown("---")
            st.markdown("#### 📊 Static vs Live Metrics Comparison")

            comparison_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Static (Test Set)': [0.86, 0.581, 0.487, 0.530],
                'Live (Current Data)': [
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0)
                ]
            })

            st.dataframe(comparison_df.style.format({'Static (Test Set)': '{:.3f}', 'Live (Current Data)': '{:.3f}'}))

            # Interpretation
            st.info("""
            **Interpretation Guide**:
            - **Similar metrics**: Model performance is consistent with test set
            - **Lower live metrics**: Current data may have different characteristics or distribution
            - **Higher live metrics**: Current data may be easier to predict (check for data quality)
            """)
    else:
        st.info("👆 Click 'Run Predictions' above to score data from the Gold table.")

# =============================
# Tab 7: Reports
# =============================
def render_reports_tab():
    st.markdown('<div class="sub-header">📑 Reports</div>', unsafe_allow_html=True)

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_reports"):
            st.cache_data.clear()
            st.rerun()

    # Summary Cards
    st.markdown("### 📊 Pipeline Summary")

    summary_data = call_api("/api/reports/summary")

    if summary_data:
        metric_cols = st.columns(5)

        with metric_cols[0]:
            st.metric("Total Runs", summary_data.get("total_runs", 0))
        with metric_cols[1]:
            latest_corrid = summary_data.get("latest_corrid", "None")
            # Truncate corrid for display
            display_corrid = latest_corrid[:20] + "..." if latest_corrid and len(latest_corrid) > 20 else latest_corrid
            st.metric("Latest CorrID", display_corrid)
            if latest_corrid and latest_corrid != "None":
                st.caption(f"Full: {latest_corrid}")
        with metric_cols[2]:
            st.metric("Gold Row Count", f"{summary_data.get('gold_row_count', 0):,}")
        with metric_cols[3]:
            latest_date = summary_data.get("latest_data_date", "N/A")
            if latest_date and latest_date != "N/A" and latest_date != "None":
                latest_date = latest_date[:10]  # Show only date
            st.metric("Latest Data Date", latest_date)
        with metric_cols[4]:
            last_run = summary_data.get("last_run_timestamp", "N/A")
            if last_run and last_run != "N/A" and last_run != "None":
                last_run = last_run[:19].replace("T", " ")  # Format datetime
            st.metric("Last Run Time", last_run)

        st.markdown("---")

    # Latest Run Summary
    st.markdown("### 🕐 Latest Run Summary")

    if summary_data and summary_data.get("latest_corrid") and summary_data.get("latest_corrid") != "None":
        latest_corrid = summary_data.get("latest_corrid")
        last_run_time = summary_data.get("last_run_timestamp", "N/A")

        run_summary = pd.DataFrame({
            "Metric": ["Correlation ID", "Run Timestamp", "Gold Rows", "Latest Crash Date"],
            "Value": [
                latest_corrid,
                last_run_time[:19].replace("T", " ") if last_run_time and last_run_time != "N/A" else "N/A",
                f"{summary_data.get('gold_row_count', 0):,}",
                summary_data.get("latest_data_date", "N/A")[:10] if summary_data.get("latest_data_date") else "N/A"
            ]
        })
        st.table(run_summary)
        st.caption("💡 Run details are extracted from MinIO storage and Gold database")
    else:
        st.info("No pipeline runs found. Start a data fetch job to see run history.")

    st.markdown("---")

    # Download Reports
    st.markdown("### 📥 Download Reports")

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        if st.button("📄 Download Run History (CSV)"):
            with st.spinner("Generating CSV report..."):
                try:
                    # Get detailed run history from API
                    history_data = call_api("/api/reports/run-history")

                    if history_data and history_data.get("runs"):
                        runs = history_data["runs"]

                        # Create DataFrame with detailed information
                        csv_data = pd.DataFrame(runs)
                        csv_data.columns = ["Correlation ID", "Run Timestamp", "Rows Processed", "Status", "Mode"]

                        # Reorder columns for better readability
                        csv_data = csv_data[["Run Timestamp", "Correlation ID", "Status", "Rows Processed", "Mode"]]
                    else:
                        # Fallback if no data found
                        csv_data = pd.DataFrame({
                            "Run Timestamp": ["No runs found"],
                            "Correlation ID": ["N/A"],
                            "Status": ["N/A"],
                            "Rows Processed": [0],
                            "Mode": ["N/A"]
                        })

                    csv = csv_data.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"run_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to generate CSV: {str(e)}")

    with download_col2:
        if st.button("📄 Download Full Report (PDF)"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Call PDF generation endpoint
                    response = requests.get(f"{API_BASE_URL}/api/reports/download/pdf", timeout=30)

                    if response.status_code == 200:
                        # Generate filename with timestamp
                        filename = f"crash_etl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                        # Provide download button
                        st.download_button(
                            label="💾 Download PDF",
                            data=response.content,
                            file_name=filename,
                            mime="application/pdf"
                        )
                        st.success("✅ PDF report generated successfully!")
                    else:
                        st.error(f"Failed to generate PDF: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

# =============================
# Main Application
# =============================
def main():
    st.sidebar.title("🚦 Navigation")
    st.sidebar.markdown("---")

    # Tab selection
    tabs = [
        "🏠 Home",
        "🧰 Data Management",
        "🔍 Data Fetcher",
        "⏰ Scheduler",
        "📊 EDA",
        "🤖 Model",
        "📑 Reports"
    ]

    selected_tab = st.sidebar.radio("Go to", tabs)

    # Track page views
    page_views.labels(page=selected_tab).inc()

    # Render selected tab
    if selected_tab == "🏠 Home":
        render_home_tab()
    elif selected_tab == "🧰 Data Management":
        render_data_management_tab()
    elif selected_tab == "🔍 Data Fetcher":
        render_data_fetcher_tab()
    elif selected_tab == "⏰ Scheduler":
        render_scheduler_tab()
    elif selected_tab == "📊 EDA":
        render_eda_tab()
    elif selected_tab == "🤖 Model":
        render_model_tab()
    elif selected_tab == "📑 Reports":
        render_reports_tab()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Chicago Crash ETL Dashboard v1.0")
    st.sidebar.caption("Built with Streamlit + FastAPI")

if __name__ == "__main__":
    main()
