"""
Scheduler Module for Chicago Crash ETL Pipeline
Manages automated scheduling of data fetch jobs using APScheduler.
"""
import os
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError

logger = logging.getLogger(__name__)

# Configuration
SCHEDULES_FILE = os.getenv("SCHEDULES_FILE", "/data/schedules/schedules.json")

# Global scheduler instance
scheduler = None
job_publisher = None  # Will be set to the publish function from server.py


class Schedule:
    """Represents a scheduled job configuration."""

    def __init__(self, schedule_id: str, cron_expr: str, config_type: str,
                 job_config: dict, enabled: bool = True, created_at: str = None,
                 last_run: str = None):
        self.id = schedule_id
        self.cron_expr = cron_expr
        self.config_type = config_type
        self.job_config = job_config
        self.enabled = enabled
        self.created_at = created_at or datetime.now().isoformat()
        self.last_run = last_run

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cron_expr": self.cron_expr,
            "config_type": self.config_type,
            "job_config": self.job_config,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run": self.last_run
        }

    @staticmethod
    def from_dict(data: dict) -> 'Schedule':
        return Schedule(
            schedule_id=data["id"],
            cron_expr=data["cron_expr"],
            config_type=data["config_type"],
            job_config=data["job_config"],
            enabled=data.get("enabled", True),
            created_at=data.get("created_at"),
            last_run=data.get("last_run")
        )


def load_schedules() -> List[Schedule]:
    """Load schedules from JSON file."""
    schedules_path = Path(SCHEDULES_FILE)

    if not schedules_path.exists():
        logger.info(f"No schedules file found at {SCHEDULES_FILE}, starting fresh")
        return []

    try:
        with open(schedules_path, 'r') as f:
            data = json.load(f)
            schedules = [Schedule.from_dict(s) for s in data]
            logger.info(f"Loaded {len(schedules)} schedules from file")
            return schedules
    except Exception as e:
        logger.error(f"Failed to load schedules: {e}")
        return []


def save_schedules(schedules: List[Schedule]):
    """Save schedules to JSON file."""
    schedules_path = Path(SCHEDULES_FILE)
    schedules_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(schedules_path, 'w') as f:
            json.dump([s.to_dict() for s in schedules], f, indent=2)
        logger.info(f"Saved {len(schedules)} schedules to file")
    except Exception as e:
        logger.error(f"Failed to save schedules: {e}")


def scheduled_job_wrapper(schedule_id: str, job_config: dict):
    """
    Wrapper function that executes when a schedule triggers.
    Updates last_run timestamp and publishes job to RabbitMQ.
    """
    global job_publisher

    logger.info(f"Schedule {schedule_id} triggered at {datetime.now().isoformat()}")

    try:
        # Update last_run timestamp
        schedules = load_schedules()
        for schedule in schedules:
            if schedule.id == schedule_id:
                schedule.last_run = datetime.now().isoformat()
                break
        save_schedules(schedules)

        # Publish job using the job publisher function
        if job_publisher:
            result = job_publisher(job_config)
            logger.info(f"Schedule {schedule_id} published job: {result}")
        else:
            logger.error("Job publisher not initialized!")

    except Exception as e:
        logger.error(f"Schedule {schedule_id} failed: {e}")


def init_scheduler(publisher_func):
    """
    Initialize the APScheduler background scheduler.
    Load existing schedules and start the scheduler.
    """
    global scheduler, job_publisher

    job_publisher = publisher_func
    scheduler = BackgroundScheduler()

    # Load and register existing schedules
    schedules = load_schedules()
    for schedule in schedules:
        if schedule.enabled:
            try:
                trigger = CronTrigger.from_crontab(schedule.cron_expr)
                scheduler.add_job(
                    scheduled_job_wrapper,
                    trigger=trigger,
                    id=schedule.id,
                    args=[schedule.id, schedule.job_config],
                    replace_existing=True
                )
                logger.info(f"Registered schedule {schedule.id}: {schedule.cron_expr}")
            except Exception as e:
                logger.error(f"Failed to register schedule {schedule.id}: {e}")

    scheduler.start()
    logger.info("Scheduler started successfully")


def add_schedule(cron_expr: str, config_type: str, job_config: dict) -> Schedule:
    """Add a new schedule."""
    global scheduler

    schedule_id = str(uuid.uuid4())[:8]
    schedule = Schedule(
        schedule_id=schedule_id,
        cron_expr=cron_expr,
        config_type=config_type,
        job_config=job_config,
        enabled=True
    )

    # Save to file
    schedules = load_schedules()
    schedules.append(schedule)
    save_schedules(schedules)

    # Add to scheduler
    try:
        trigger = CronTrigger.from_crontab(cron_expr)
        scheduler.add_job(
            scheduled_job_wrapper,
            trigger=trigger,
            id=schedule_id,
            args=[schedule_id, job_config],
            replace_existing=True
        )
        logger.info(f"Added schedule {schedule_id}: {cron_expr}")
    except Exception as e:
        logger.error(f"Failed to add schedule to APScheduler: {e}")
        raise

    return schedule


def remove_schedule(schedule_id: str) -> bool:
    """Remove a schedule."""
    global scheduler

    # Remove from scheduler
    try:
        scheduler.remove_job(schedule_id)
    except JobLookupError:
        logger.warning(f"Schedule {schedule_id} not found in APScheduler")

    # Remove from file
    schedules = load_schedules()
    schedules = [s for s in schedules if s.id != schedule_id]
    save_schedules(schedules)

    logger.info(f"Removed schedule {schedule_id}")
    return True


def toggle_schedule(schedule_id: str, enabled: bool) -> Optional[Schedule]:
    """Enable or disable a schedule."""
    global scheduler

    schedules = load_schedules()
    target_schedule = None

    for schedule in schedules:
        if schedule.id == schedule_id:
            schedule.enabled = enabled
            target_schedule = schedule
            break

    if not target_schedule:
        return None

    save_schedules(schedules)

    # Update APScheduler
    if enabled:
        # Add/resume job
        try:
            trigger = CronTrigger.from_crontab(target_schedule.cron_expr)
            scheduler.add_job(
                scheduled_job_wrapper,
                trigger=trigger,
                id=schedule_id,
                args=[schedule_id, target_schedule.job_config],
                replace_existing=True
            )
            logger.info(f"Enabled schedule {schedule_id}")
        except Exception as e:
            logger.error(f"Failed to enable schedule: {e}")
            raise
    else:
        # Pause/remove job
        try:
            scheduler.remove_job(schedule_id)
            logger.info(f"Disabled schedule {schedule_id}")
        except JobLookupError:
            logger.warning(f"Schedule {schedule_id} not found in APScheduler")

    return target_schedule


def get_schedules() -> List[Schedule]:
    """Get all schedules."""
    return load_schedules()


def shutdown_scheduler():
    """Shutdown the scheduler gracefully."""
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down")
