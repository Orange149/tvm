"""Observe frozen fluid-model tasks without changing their service or scheduling."""
from model_semantics_audit.simulator import AuditReplay
from thread_history_audit.simulator import HistoryReplay


class TraceReplay(HistoryReplay):
    def _close_task(self,key):
        # HistoryReplay suppresses individual task storage to reduce old files.
        # Restore it without changing event order or scheduling decisions.
        return AuditReplay._close_task(self,key)
