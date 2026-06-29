class AppealError(Exception):
    pass


def process_appeal(content_id: str, creator_id: str, reason: str, logger) -> dict:
    """Validate and process a creator appeal for a content classification."""
    if not reason or not reason.strip():
        raise AppealError("reason must not be empty")

    entry = logger.get_entry(content_id)
    if entry is None:
        raise AppealError(f"content_id {content_id} not found")

    if entry["creator_id"] != creator_id:
        raise AppealError("creator_id does not match the original submission")

    appeal_id = logger.log_appeal(
        content_id=content_id,
        creator_id=creator_id,
        reason=reason.strip(),
    )

    return {
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": (
            "Your appeal has been received. "
            "The classification has been marked as under review."
        ),
    }
