"""
AgentCircuit Root Cause Analysis (RCA) Module

Provides automated analysis of failures to identify patterns,
root causes, and actionable recommendations.
"""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .storage import Storage
from .errors import ErrorCategory, ErrorClassifier, ClassifiedError


class RCACategory(Enum):
    """High-level root cause categories."""
    INPUT_QUALITY = "input_quality"
    SCHEMA_MISMATCH = "schema_mismatch"
    LLM_BEHAVIOR = "llm_behavior"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    RESOURCE_LIMITS = "resource_limits"
    LOGIC_ERROR = "logic_error"
    UNKNOWN = "unknown"


@dataclass
class RootCause:
    """A identified root cause with details."""
    category: RCACategory
    description: str
    confidence: float  # 0.0 to 1.0
    affected_nodes: List[str]
    occurrence_count: int
    first_seen: str
    last_seen: str
    sample_errors: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "confidence": self.confidence,
            "affected_nodes": self.affected_nodes,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sample_errors": self.sample_errors[:5],
            "recommendations": self.recommendations
        }


@dataclass
class RCAReport:
    """Complete root cause analysis report."""
    analysis_id: str
    generated_at: datetime
    time_range: Tuple[str, str]
    total_failures: int
    total_runs: int
    failure_rate: float
    root_causes: List[RootCause]
    patterns: List[Dict[str, Any]]
    trending_issues: List[Dict[str, Any]]
    recommendations: List[str]

    @property
    def top_cause(self) -> Optional[RootCause]:
        """Get the most common root cause."""
        if self.root_causes:
            return max(self.root_causes, key=lambda x: x.occurrence_count)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "generated_at": self.generated_at.isoformat(),
            "time_range": {
                "start": self.time_range[0],
                "end": self.time_range[1]
            },
            "summary": {
                "total_failures": self.total_failures,
                "total_runs": self.total_runs,
                "failure_rate": round(self.failure_rate * 100, 2)
            },
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "patterns": self.patterns,
            "trending_issues": self.trending_issues,
            "recommendations": self.recommendations
        }


class RootCauseAnalyzer:
    """
    Analyzes failures to identify root causes and patterns.

    Uses historical data, error classification, and pattern matching
    to determine why failures are occurring and how to fix them.

    Usage:
        analyzer = RootCauseAnalyzer()

        # Generate a full report
        report = analyzer.analyze(hours=24)
        print(f"Top cause: {report.top_cause.description}")

        # Analyze a specific node
        causes = analyzer.analyze_node("my_node")

        # Get recommendations
        recs = analyzer.get_recommendations(report)
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the analyzer.

        Args:
            db_path: Path to the database. Uses default if not specified.
        """
        self.storage = Storage(db_path=db_path)
        self.classifier = ErrorClassifier()

        # Pattern definitions for root cause identification
        self._patterns = {
            RCACategory.INPUT_QUALITY: [
                "missing required field",
                "invalid input",
                "null value",
                "empty string",
                "malformed",
                "encoding error"
            ],
            RCACategory.SCHEMA_MISMATCH: [
                "validation error",
                "schema",
                "type error",
                "expected",
                "sentinel alert",
                "pydantic"
            ],
            RCACategory.LLM_BEHAVIOR: [
                "refused",
                "cannot",
                "i'm sorry",
                "i can't",
                "inappropriate",
                "json",
                "parse",
                "format"
            ],
            RCACategory.EXTERNAL_SERVICE: [
                "timeout",
                "connection",
                "network",
                "api error",
                "rate limit",
                "503",
                "502",
                "500"
            ],
            RCACategory.RESOURCE_LIMITS: [
                "context length",
                "token limit",
                "memory",
                "quota",
                "exceeded",
                "too long"
            ],
            RCACategory.CONFIGURATION: [
                "api key",
                "authentication",
                "permission",
                "config",
                "environment",
                "not found"
            ],
            RCACategory.LOGIC_ERROR: [
                "loop detected",
                "infinite",
                "recursion",
                "deadlock",
                "assertion"
            ]
        }

        # Recommendations for each category
        self._recommendations = {
            RCACategory.INPUT_QUALITY: [
                "Add input validation before the failing node",
                "Implement data sanitization for edge cases",
                "Add default values for optional fields",
                "Log and monitor input quality metrics"
            ],
            RCACategory.SCHEMA_MISMATCH: [
                "Review and update the output schema",
                "Add more flexible schema validation with optional fields",
                "Implement schema migration for evolving outputs",
                "Use LLM repair strategies for minor schema violations"
            ],
            RCACategory.LLM_BEHAVIOR: [
                "Refine prompts to get consistent JSON output",
                "Add explicit output format instructions",
                "Use structured output features if available",
                "Implement retry with reformulated prompts"
            ],
            RCACategory.EXTERNAL_SERVICE: [
                "Implement exponential backoff for retries",
                "Add circuit breaker pattern for external calls",
                "Configure appropriate timeouts",
                "Set up fallback providers"
            ],
            RCACategory.RESOURCE_LIMITS: [
                "Implement context truncation strategy",
                "Split large inputs into chunks",
                "Use summarization for long contexts",
                "Monitor and alert on token usage"
            ],
            RCACategory.CONFIGURATION: [
                "Verify API keys and credentials",
                "Check environment variable configuration",
                "Review permission settings",
                "Add configuration validation at startup"
            ],
            RCACategory.LOGIC_ERROR: [
                "Review loop detection thresholds",
                "Add state tracking to prevent cycles",
                "Implement escape conditions for recursive flows",
                "Use Fuse with appropriate limits"
            ],
            RCACategory.UNKNOWN: [
                "Enable detailed error logging",
                "Add more context to error messages",
                "Review recent code changes",
                "Check for environmental factors"
            ]
        }

    def analyze(
        self,
        hours: int = 24,
        node_id: Optional[str] = None
    ) -> RCAReport:
        """
        Perform root cause analysis on recent failures.

        Args:
            hours: Number of hours to analyze
            node_id: Optional specific node to analyze

        Returns:
            RCAReport with findings and recommendations
        """
        import secrets

        # Get failure data
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        traces = self._get_traces(cutoff, node_id)

        failures = [t for t in traces if t.get("status", "").startswith("failed")]
        all_traces = traces

        if not failures:
            return RCAReport(
                analysis_id=secrets.token_hex(8),
                generated_at=datetime.now(),
                time_range=(cutoff, datetime.now().isoformat()),
                total_failures=0,
                total_runs=len(all_traces),
                failure_rate=0.0,
                root_causes=[],
                patterns=[],
                trending_issues=[],
                recommendations=["No failures detected in the analysis period."]
            )

        # Identify root causes
        root_causes = self._identify_root_causes(failures)

        # Find patterns
        patterns = self._find_patterns(failures)

        # Identify trending issues
        trending = self._find_trending(failures, hours)

        # Generate recommendations
        recommendations = self._generate_recommendations(root_causes, patterns)

        failure_rate = len(failures) / len(all_traces) if all_traces else 0

        return RCAReport(
            analysis_id=secrets.token_hex(8),
            generated_at=datetime.now(),
            time_range=(cutoff, datetime.now().isoformat()),
            total_failures=len(failures),
            total_runs=len(all_traces),
            failure_rate=failure_rate,
            root_causes=root_causes,
            patterns=patterns,
            trending_issues=trending,
            recommendations=recommendations
        )

    def analyze_node(self, node_id: str, hours: int = 24) -> RCAReport:
        """Analyze a specific node's failures."""
        return self.analyze(hours=hours, node_id=node_id)

    def analyze_run(self, run_id: str) -> Dict[str, Any]:
        """
        Analyze a specific run to understand what went wrong.

        Args:
            run_id: Run ID to analyze

        Returns:
            Detailed analysis of the run
        """
        history = self.storage.get_run_history(run_id)
        if not history:
            return {"error": f"Run '{run_id}' not found"}

        analysis = {
            "run_id": run_id,
            "total_steps": len(history),
            "timeline": [],
            "failures": [],
            "repairs": [],
            "root_causes": [],
            "recommendations": []
        }

        for trace in history:
            step = {
                "node_id": trace.get("node_id"),
                "status": trace.get("status"),
                "timestamp": trace.get("timestamp"),
                "duration_ms": trace.get("duration_ms")
            }
            analysis["timeline"].append(step)

            status = trace.get("status", "")
            if status.startswith("failed"):
                diagnosis = trace.get("diagnosis", "")
                category = self._categorize_error(diagnosis)

                failure = {
                    "node_id": trace.get("node_id"),
                    "status": status,
                    "diagnosis": diagnosis,
                    "category": category.value,
                    "recommendations": self._recommendations.get(category, [])[:2]
                }
                analysis["failures"].append(failure)

                # Add root cause
                if category not in [rc["category"] for rc in analysis["root_causes"]]:
                    analysis["root_causes"].append({
                        "category": category.value,
                        "description": self._get_category_description(category),
                        "recommendations": self._recommendations.get(category, [])
                    })

            elif status == "repaired":
                analysis["repairs"].append({
                    "node_id": trace.get("node_id"),
                    "recovery_attempts": trace.get("recovery_attempts", 0),
                    "diagnosis": trace.get("diagnosis")
                })

        # Generate overall recommendations
        if analysis["root_causes"]:
            for rc in analysis["root_causes"]:
                analysis["recommendations"].extend(rc["recommendations"][:2])
        analysis["recommendations"] = list(set(analysis["recommendations"]))[:5]

        return analysis

    def get_node_health(self, node_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get health metrics for a specific node.

        Args:
            node_id: Node to analyze
            hours: Time window

        Returns:
            Health metrics and status
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        traces = self._get_traces(cutoff, node_id)

        if not traces:
            return {
                "node_id": node_id,
                "status": "unknown",
                "message": "No data available"
            }

        total = len(traces)
        success = sum(1 for t in traces if t.get("status") == "success")
        repaired = sum(1 for t in traces if t.get("status") == "repaired")
        failed = sum(1 for t in traces if t.get("status", "").startswith("failed"))

        success_rate = (success + repaired) / total if total > 0 else 0
        avg_duration = sum(t.get("duration_ms", 0) or 0 for t in traces) / total if total > 0 else 0

        # Determine health status
        if success_rate >= 0.95:
            status = "healthy"
        elif success_rate >= 0.80:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "node_id": node_id,
            "status": status,
            "metrics": {
                "total_executions": total,
                "success_count": success,
                "repaired_count": repaired,
                "failed_count": failed,
                "success_rate": round(success_rate * 100, 2),
                "avg_duration_ms": round(avg_duration, 2)
            },
            "time_range": {
                "start": cutoff,
                "end": datetime.now().isoformat()
            }
        }

    def _get_traces(
        self,
        cutoff: str,
        node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get traces from storage with filtering."""
        traces = self.storage.get_traces(limit=10000)

        # Filter by time
        filtered = [
            t for t in traces
            if t.get("timestamp", "") >= cutoff
        ]

        # Filter by node if specified
        if node_id:
            filtered = [t for t in filtered if t.get("node_id") == node_id]

        return filtered

    def _identify_root_causes(
        self,
        failures: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Identify root causes from failures."""
        # Group failures by category
        categorized = defaultdict(list)

        for failure in failures:
            diagnosis = failure.get("diagnosis", "") or ""
            error_category = failure.get("error_category", "")

            # Determine RCA category
            rca_category = self._categorize_error(diagnosis, error_category)
            categorized[rca_category].append(failure)

        # Create root causes
        root_causes = []
        for category, category_failures in categorized.items():
            if not category_failures:
                continue

            # Get affected nodes
            affected_nodes = list(set(f.get("node_id", "") for f in category_failures))

            # Get timestamps
            timestamps = [f.get("timestamp", "") for f in category_failures]
            timestamps = [t for t in timestamps if t]
            first_seen = min(timestamps) if timestamps else ""
            last_seen = max(timestamps) if timestamps else ""

            # Get sample errors
            sample_errors = [
                f.get("diagnosis", "")[:200]
                for f in category_failures[:5]
                if f.get("diagnosis")
            ]

            # Calculate confidence based on pattern match strength
            confidence = min(0.95, 0.5 + (len(category_failures) / len(failures)) * 0.5)

            root_cause = RootCause(
                category=category,
                description=self._get_category_description(category),
                confidence=confidence,
                affected_nodes=affected_nodes,
                occurrence_count=len(category_failures),
                first_seen=first_seen,
                last_seen=last_seen,
                sample_errors=sample_errors,
                recommendations=self._recommendations.get(category, [])
            )
            root_causes.append(root_cause)

        # Sort by occurrence count
        root_causes.sort(key=lambda x: x.occurrence_count, reverse=True)
        return root_causes

    def _categorize_error(
        self,
        diagnosis: str,
        error_category: str = ""
    ) -> RCACategory:
        """Categorize an error into an RCA category."""
        diagnosis_lower = diagnosis.lower()

        # Check error category first
        if error_category:
            ec_mapping = {
                "json_parse": RCACategory.LLM_BEHAVIOR,
                "schema_validation": RCACategory.SCHEMA_MISMATCH,
                "network_timeout": RCACategory.EXTERNAL_SERVICE,
                "rate_limit": RCACategory.EXTERNAL_SERVICE,
                "auth_failure": RCACategory.CONFIGURATION,
                "context_length": RCACategory.RESOURCE_LIMITS,
                "llm_refusal": RCACategory.LLM_BEHAVIOR,
                "tool_error": RCACategory.LOGIC_ERROR
            }
            if error_category in ec_mapping:
                return ec_mapping[error_category]

        # Pattern matching on diagnosis
        for category, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern in diagnosis_lower:
                    return category

        return RCACategory.UNKNOWN

    def _get_category_description(self, category: RCACategory) -> str:
        """Get a human-readable description for a category."""
        descriptions = {
            RCACategory.INPUT_QUALITY: "Input data quality issues causing validation or processing failures",
            RCACategory.SCHEMA_MISMATCH: "Output schema validation failures from unexpected LLM responses",
            RCACategory.LLM_BEHAVIOR: "LLM behavior issues including refusals, format errors, or inconsistent outputs",
            RCACategory.EXTERNAL_SERVICE: "External service failures including timeouts, rate limits, or API errors",
            RCACategory.RESOURCE_LIMITS: "Resource limit issues including context length or memory constraints",
            RCACategory.CONFIGURATION: "Configuration issues including missing credentials or incorrect settings",
            RCACategory.LOGIC_ERROR: "Logic errors including loops, recursion, or deadlock conditions",
            RCACategory.UNKNOWN: "Unknown or unclassified error conditions"
        }
        return descriptions.get(category, "Unknown category")

    def _find_patterns(
        self,
        failures: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find patterns in failures."""
        patterns = []

        # Node failure pattern
        node_failures = Counter(f.get("node_id") for f in failures)
        if node_failures:
            top_node, count = node_failures.most_common(1)[0]
            if count > 2:
                patterns.append({
                    "type": "node_concentration",
                    "description": f"Node '{top_node}' accounts for {count} failures ({count / len(failures) * 100:.0f}%)",
                    "node_id": top_node,
                    "count": count,
                    "severity": "high" if count / len(failures) > 0.5 else "medium"
                })

        # Time-based pattern
        hour_buckets = Counter()
        for f in failures:
            ts = f.get("timestamp", "")
            if " " in ts:
                hour = ts.split(" ")[1][:2]
                hour_buckets[hour] += 1

        if hour_buckets:
            top_hour, count = hour_buckets.most_common(1)[0]
            if count > len(failures) * 0.3:
                patterns.append({
                    "type": "time_concentration",
                    "description": f"Failures concentrated around hour {top_hour}:00",
                    "hour": top_hour,
                    "count": count,
                    "severity": "medium"
                })

        # Sequential failure pattern
        sequential_count = 0
        last_run = None
        for f in sorted(failures, key=lambda x: x.get("timestamp", "")):
            run_id = f.get("run_id")
            if run_id == last_run:
                sequential_count += 1
            last_run = run_id

        if sequential_count > 3:
            patterns.append({
                "type": "cascading_failures",
                "description": f"Detected {sequential_count} cascading failures within single runs",
                "count": sequential_count,
                "severity": "high"
            })

        return patterns

    def _find_trending(
        self,
        failures: List[Dict[str, Any]],
        hours: int
    ) -> List[Dict[str, Any]]:
        """Find trending (increasing) issues."""
        trending = []

        # Split into two halves
        mid_time = (datetime.now() - timedelta(hours=hours / 2)).isoformat()

        first_half = [f for f in failures if f.get("timestamp", "") < mid_time]
        second_half = [f for f in failures if f.get("timestamp", "") >= mid_time]

        # Compare node failures
        first_nodes = Counter(f.get("node_id") for f in first_half)
        second_nodes = Counter(f.get("node_id") for f in second_half)

        for node, count in second_nodes.items():
            first_count = first_nodes.get(node, 0)
            if count > first_count * 1.5 and count > 2:
                trending.append({
                    "node_id": node,
                    "trend": "increasing",
                    "first_half_count": first_count,
                    "second_half_count": count,
                    "change_percent": round(((count - first_count) / max(first_count, 1)) * 100)
                })

        return trending

    def _generate_recommendations(
        self,
        root_causes: List[RootCause],
        patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Add recommendations from top root causes
        for rc in root_causes[:3]:
            recommendations.extend(rc.recommendations[:2])

        # Add pattern-specific recommendations
        for pattern in patterns:
            if pattern["type"] == "node_concentration":
                recommendations.append(
                    f"Prioritize fixing node '{pattern['node_id']}' which has the highest failure rate"
                )
            elif pattern["type"] == "cascading_failures":
                recommendations.append(
                    "Implement circuit breaker pattern to prevent cascading failures"
                )
            elif pattern["type"] == "time_concentration":
                recommendations.append(
                    f"Investigate system conditions around {pattern['hour']}:00 for potential issues"
                )

        # Deduplicate and limit
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        return unique_recs[:10]


# Convenience function
def analyze_failures(hours: int = 24, db_path: Optional[str] = None) -> RCAReport:
    """
    Quick function to analyze recent failures.

    Args:
        hours: Number of hours to analyze
        db_path: Optional database path

    Returns:
        RCAReport with findings
    """
    analyzer = RootCauseAnalyzer(db_path=db_path)
    return analyzer.analyze(hours=hours)
