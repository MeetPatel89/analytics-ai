"""Tools for incident response."""

import json


def get_server_health(server_id: str) -> str:
    """Get CPU and Memory usage for a given server."""
    metrics = {
        # Scenario 1: High CPU (Needs Restart)
        "payment-server-01": {"cpu": "98%", "memory": "40%", "status": "Warning"},
        # Scenario 2: Healthy (No Action Needed)
        "db-node-02": {"cpu": "12%", "memory": "60%", "status": "Healthy"},
        # Scenario 3: High Memory Leak (Needs Restart or Escalation)
        "auth-service-03": {"cpu": "45%", "memory": "95%", "status": "Critical"},
        # Scenario 4: Network/Dependency Failure (Needs Escalation)
        "search-index-09": {"cpu": "10%", "memory": "15%", "status": "Error"},
        # Scenario 5: Completely Normal
        "frontend-node-04": {"cpu": "25%", "memory": "30%", "status": "Healthy"},
    }

    result = metrics.get(server_id, {"error": "Server not found. Check the ID."})
    return json.dumps(result)


def fetch_recent_logs(server_id: str, lines: int = 5) -> str:
    """Fetch the last N lines of logs for a given server."""
    # Different logs for different servers to trigger different agent behaviors
    log_database = {
        "payment-server-01": [
            "[INFO] Request received /pay/v1",
            "[WARN] CPU threshold exceeded 90%",
            "[WARN] Thread pool exhaustion",
            "[CRITICAL] Process hung, not accepting new connections",
            "[ERROR] Timeout waiting for thread",
        ],
        "db-node-02": [
            "[INFO] Backup started",
            "[INFO] Backup completed successfully",
            "[INFO] User query executed in 12ms",
            "[INFO] Health check: OK",
            "[INFO] Replication sync active",
        ],
        "auth-service-03": [
            "[INFO] Token validated user_882",
            "[WARN] Garbage collection taking too long (>5s)",
            "[ERROR] java.lang.OutOfMemoryError: Java heap space",
            "[CRITICAL] Application crashing due to memory leak",
            "[INFO] Restarting context...",
        ],
        "search-index-09": [
            "[INFO] Indexing started",
            "[ERROR] Connection refused: elastic-cluster-main:9200",
            "[ERROR] Failed to write document ID 4432",
            "[CRITICAL] Dependency Unreachable: Search Engine is down",
            "[ERROR] Retrying in 30s...",
        ],
        "frontend-node-04": [
            "[INFO] GET /home 200 OK",
            "[INFO] GET /assets/logo.png 200 OK",
            "[INFO] GET /login 200 OK",
            "[INFO] GET /api/v1/status 200 OK",
            "[INFO] Health check passed",
        ],
    }

    # Default logs if server not found in specific list
    default_logs = ["[INFO] System stable", "[INFO] Heartbeat signal received"]

    logs = log_database.get(server_id, default_logs)
    return json.dumps({"logs": logs[:lines]})


def restart_service(server_id: str) -> str:
    """Restart a given service."""
    return json.dumps(
        {
            "status": "success",
            "message": f"Service {server_id} restarted successfully.",
        }
    )


def escalate_incident(server_id: str) -> str:
    """Escalate an incident to a higher level of support."""
    return json.dumps(
        {
            "status": "success",
            "message": f"Incident for {server_id} escalated successfully.",
        }
    )
