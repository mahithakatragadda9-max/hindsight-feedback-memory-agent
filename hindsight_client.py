import requests

from config import HINDSIGHT_API_URL, HINDSIGHT_API_KEY, HINDSIGHT_BANK_ID
from models import Memory


class HindsightUnavailable(Exception):
    pass


class HindsightClient:
    """Hindsight Cloud REST adapter with workspace-isolated memory."""

    def __init__(self):
        self.base_url = HINDSIGHT_API_URL.rstrip("/") if HINDSIGHT_API_URL else ""
        self.api_key = HINDSIGHT_API_KEY
        self.bank_id = HINDSIGHT_BANK_ID
        self.enabled = bool(
            self.base_url
            and self.api_key
            and self.bank_id
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _ensure(self) -> None:
        if not self.enabled:
            raise HindsightUnavailable(
                "Hindsight credentials are not configured "
                "(HINDSIGHT_API_URL / HINDSIGHT_API_KEY / HINDSIGHT_BANK_ID)."
            )

    def _workspace_tag(self, workspace: str) -> str:
        """Create one stable Hindsight tag for a workspace."""
        return f"workspace:{workspace.strip()}"

    def retain(self, memory: Memory) -> None:
        """Store one feedback experience in Hindsight."""
        self._ensure()

        workspace = memory.workspace.strip()
        workspace_tag = self._workspace_tag(workspace)

        content = (
            f"Workspace: {workspace}\n"
            f"Feedback ID: {memory.feedback_id}\n"
            f"Channel: {memory.channel}\n"
            f"Customer feedback: {memory.raw_text}\n"
            f"Theme: {memory.discovered_theme}\n"
            f"Underlying issue: {memory.underlying_issue}\n"
            f"Failure mode: {memory.failure_mode}\n"
            f"Sentiment: {memory.sentiment}\n"
            f"Priority: {memory.priority}\n"
            f"Agent interpretation: {memory.agent_interpretation}\n"
            f"Lifecycle state: {memory.lifecycle_state}"
        )

        payload = {
            "items": [
                {
                    "content": content,

                    "context": (
                        "feedback_memory_agent | "
                        f"workspace={workspace} | "
                        f"channel={memory.channel}"
                    ),

                    # Explicit workspace metadata.
                    "metadata": {
                        "workspace": workspace,
                        "feedback_id": str(memory.feedback_id),
                        "channel": str(memory.channel),
                    },

                    # Critical: Hindsight visibility scope.
                    "tags": [workspace_tag],

                    **(
                        {"timestamp": memory.timestamp}
                        if memory.timestamp
                        else {}
                    ),
                }
            ]
        }

        url = (
            f"{self.base_url}/v1/default/banks/"
            f"{self.bank_id}/memories"
        )

        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )

            if not response.ok:
                raise HindsightUnavailable(
                    f"Hindsight retain failed "
                    f"(HTTP {response.status_code}): "
                    f"{response.text[:1000]}"
                )

        except requests.RequestException as exc:
            raise HindsightUnavailable(
                f"Hindsight retain request failed: {exc}"
            ) from exc

    def recall(
        self,
        query: str,
        workspace: str,
        top_k: int = 20,
    ) -> list[Memory]:
        """Recall only experiences belonging to the requested workspace."""
        self._ensure()

        workspace = workspace.strip()
        workspace_tag = self._workspace_tag(workspace)

        payload = {
            # Isolation is enforced by the tag filter below, never by the wording
            # of the query, so the internal workspace id is kept out of it.
            "query": (
                "Find previous customer experiences relevant to this feedback.\n"
                f"Current feedback context: {query}"
            ),


            # Critical workspace isolation.
            "tags": [workspace_tag],
            "tags_match": "any_strict",

            "budget": "mid",
            "max_tokens": max(
                2000,
                min(top_k * 500, 8000),
            ),
        }

        url = (
            f"{self.base_url}/v1/default/banks/"
            f"{self.bank_id}/memories/recall"
        )

        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )

            if not response.ok:
                raise HindsightUnavailable(
                    f"Hindsight recall failed "
                    f"(HTTP {response.status_code}): "
                    f"{response.text[:1000]}"
                )

            data = response.json()

        except requests.RequestException as exc:
            raise HindsightUnavailable(
                f"Hindsight recall request failed: {exc}"
            ) from exc

        results = (
            data.get("results")
            or data.get("memories")
            or []
        )

        output: list[Memory] = []

        for item in results[:top_k]:
            try:
                memory = self._convert_result_to_memory(
                    item,
                    workspace,
                )

                # Never accept an unscoped or mismatched memory.
                if memory is None:
                    continue

                if memory.workspace.strip() != workspace:
                    continue

                output.append(memory)

            except Exception:
                # Ignore malformed Hindsight results.
                continue

        return output

    def _convert_result_to_memory(
        self,
        item: dict,
        workspace: str,
    ) -> Memory | None:
        """
        Convert a Hindsight result into the app's Memory model.

        Workspace is taken from explicit Hindsight metadata first,
        then from structured content.

        IMPORTANT:
        If the result contains no verifiable workspace,
        it is rejected instead of being assigned to the current workspace.
        """

        memory_id = str(item.get("id", ""))

        text = (
            item.get("text")
            or item.get("content")
            or ""
        )

        mentioned_at = (
            item.get("mentioned_at")
            or item.get("timestamp")
            or ""
        )

        parsed = self._parse_structured_content(text)

        # Hindsight metadata is the strongest workspace signal.
        metadata = item.get("metadata") or {}

        metadata_workspace = metadata.get("workspace")

        content_workspace = parsed.get("Workspace")

        result_workspace = (
            metadata_workspace
            or content_workspace
        )

        # NEVER assume missing workspace == current workspace.
        if not result_workspace:
            return None

        result_workspace = str(result_workspace).strip()

        # Hard workspace isolation.
        if result_workspace != workspace.strip():
            return None

        return Memory(
            memory_id=memory_id,

            feedback_id=parsed.get(
                "Feedback ID",
                metadata.get("feedback_id", ""),
            ),

            workspace=result_workspace,

            timestamp=mentioned_at,

            channel=parsed.get(
                "Channel",
                metadata.get("channel", ""),
            ),

            raw_text=parsed.get(
                "Customer feedback",
                text,
            ),

            discovered_theme=parsed.get(
                "Theme",
                "",
            ),

            underlying_issue=parsed.get(
                "Underlying issue",
                "",
            ),

            sentiment=parsed.get(
                "Sentiment",
                "",
            ),

            priority=parsed.get(
                "Priority",
                "",
            ),

            agent_interpretation=parsed.get(
                "Agent interpretation",
                "",
            ),

            affected_capability="",

            failure_mode=parsed.get(
                "Failure mode",
                "",
            ),

            lifecycle_state=parsed.get(
                "Lifecycle state",
                "NOVEL",
            ),

            product_change_id=None,
            expected_outcome=None,
            observed_outcome=None,
            recurrence_info=None,
            evidence_used=[],

            metadata={
                "hindsight_type": item.get(
                    "type",
                    "",
                ),
                "entities": item.get(
                    "entities",
                    [],
                ),
                "workspace": result_workspace,
                "workspace_tag": self._workspace_tag(
                    result_workspace
                ),
            },
        )

    @staticmethod
    def _parse_structured_content(
        text: str,
    ) -> dict[str, str]:
        """Parse the structured memory text."""
        parsed: dict[str, str] = {}

        for line in text.splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            # The "Workspace" header is written first by retain(); a later line of
            # free-text feedback that mimics it must never override it.
            if key == "Workspace" and key in parsed:
                continue

            if key:
                parsed[key] = value

        return parsed