import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import create_model, BaseModel
from crewai.tools import BaseTool
from utils.logger import get_logger

logger = get_logger("connectors.base")


class BaseConnector:
    """
    Base class for all enterprise connectors.
    Handles auth, retries, audit logging, and wrapping methods as CrewAI tools.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    def authenticate(self) -> None:
        """Override in subclasses to set up credentials/sessions."""
        pass

    def execute(self, action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self.retry(action, *args, **kwargs)

    def retry(self, fn: Callable[..., Any], *args: Any, max_retries: int = 3, **kwargs: Any) -> Any:
        """Exponential backoff retry wrapper."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Connector {self.name} attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2.0

    def audit(
        self,
        action: str,
        params: Dict[str, Any],
        status: str,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            from monitoring.audit import log_audit_record
            log_audit_record({
                "event_type": "connector_call",
                "connector_name": self.name,
                "action": action,
                "params": params,
                "status": status,
                "result": str(result) if result is not None else None,
                "error": error,
            })
        except Exception as e:
            logger.debug(f"Audit log write skipped: {e}")

    def as_crewai_tools(self) -> List[BaseTool]:
        """
        Reflects on all public methods of the subclass and returns each one
        wrapped as a CrewAI BaseTool.
        """
        tools = []
        base_methods = set(dir(BaseConnector))

        for attr_name in dir(self):
            if attr_name.startswith("_") or attr_name in base_methods:
                continue
            member = getattr(self, attr_name)
            if not inspect.ismethod(member):
                continue
            tools.append(_make_tool(self, attr_name, member))

        return tools


# ---------------------------------------------------------------------------
# Module-level factory — avoids Pydantic v2 class-body closure issues
# ---------------------------------------------------------------------------

def _make_tool(connector: BaseConnector, method_name: str, method: Callable) -> BaseTool:
    """
    Creates a concrete BaseTool for a single connector method.

    Pydantic v2 validates class bodies at class-definition time, which means
    local variables from an enclosing function are not available as field
    defaults.  The standard workaround is to define the class at module level
    and inject the varying data (schema, name, description) via a closure
    captured in the _run override — but CrewAI/Pydantic still validates the
    declared field types at import time.

    The cleanest solution: use a plain wrapper object that inherits BaseTool
    and overrides only what changes per-method, captured through __init__.
    """
    # Build the argument schema from the method signature
    sig = inspect.signature(method)
    fields: Dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        ann = param.annotation if param.annotation != inspect.Parameter.empty else Any
        default = param.default if param.default != inspect.Parameter.empty else ...
        fields[param_name] = (ann, default)

    schema_cls: Type[BaseModel] = create_model(
        f"{connector.name}_{method_name}_schema", **fields
    )

    tool_name = f"{connector.name}.{method_name}"
    tool_desc = (method.__doc__ or f"Execute {tool_name}.").strip()

    class _ConnectorTool(BaseTool):
        # These are set as CLASS attributes so Pydantic sees them at class
        # creation time.  Each call to _make_tool produces a new class via the
        # closure so the values are unique per tool.
        name: str = tool_name          # type: ignore[assignment]
        description: str = tool_desc  # type: ignore[assignment]
        args_schema: Type[BaseModel] = schema_cls  # type: ignore[assignment]

        def _run(self, **kwargs: Any) -> str:  # type: ignore[override]
            # ── resolve calling agent for RBAC ──────────────────────────────
            agent_role = "unknown"
            for frame in inspect.stack():
                frame_self = frame.frame.f_locals.get("self")
                if frame_self and frame_self.__class__.__name__ == "Agent":
                    agent_role = getattr(frame_self, "role", None) or getattr(
                        frame_self, "id", "unknown"
                    )
                    break

            normalized = agent_role.lower().replace(" ", "_")

            # ── RBAC ────────────────────────────────────────────────────────
            try:
                from security.rbac import check_permission
                check_permission(normalized, tool_name)
            except PermissionError as pe:
                connector.audit(method_name, kwargs, "rbac_denied", error=str(pe))
                return f"Permission Denied: {pe}"
            except Exception as e:
                logger.warning(f"RBAC check non-blocking error: {e}")

            # ── execute ─────────────────────────────────────────────────────
            try:
                connector.authenticate()
                result = connector.execute(method, **kwargs)
                connector.audit(method_name, kwargs, "success", result=result)
                return str(result)
            except Exception as e:
                connector.audit(method_name, kwargs, "failed", error=str(e))
                return f"Error executing {tool_name}: {e}"

    # Give each dynamically created class a unique __qualname__ so Pydantic's
    # model registry doesn't complain about duplicate model names.
    _ConnectorTool.__name__ = f"Tool_{connector.name}_{method_name}"
    _ConnectorTool.__qualname__ = f"Tool_{connector.name}_{method_name}"

    return _ConnectorTool()
