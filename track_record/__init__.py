"""
track_record — live pre-game pick publication and result tracking.

Usage:
    from track_record.db import TrackRecordDB
    from track_record.publisher import publish_daily_picks
    from track_record.reconciler import reconcile_pending
    from track_record.stats import compute_stats
"""

from track_record.db import TrackRecordDB

__all__ = ["TrackRecordDB"]
