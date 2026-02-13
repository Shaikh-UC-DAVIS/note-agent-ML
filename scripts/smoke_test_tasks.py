from ml.extraction import LLMExtractor
from backend.storage import LocalTaskStore
from ml.intelligence import IntelligenceLayer
from ml.graph import KnowledgeGraph
from ml.search import HybridSearchEngine

def main():
    # 1) Extract
    extractor = LLMExtractor()
    text = """
    [ ] Finish project report by Feb 20
    Submit slides next Monday at 3 PM
    Call mentor tomorrow
    """

    result = extractor.extract(
        text=text,
        note_id="note-123",
        user_id="user-42",
        workspace_id="ws-9"
    )

    # 2) Persist tasks
    store = LocalTaskStore(base_path="./data")
    task_records = []
    for obj in result.objects:
        if obj.type == "Task":
            attrs = obj.attributes or {}
            task_records.append({
                "id": obj.id,
                "canonical_text": obj.canonical_text,
                "title": obj.canonical_text,
                "confidence": obj.confidence,
                "note_id": attrs.get("note_id"),
                "user_id": attrs.get("user_id"),
                "workspace_id": attrs.get("workspace_id"),
                "due_date": attrs.get("due_date"),
                "due_time": attrs.get("due_time"),
                "status": attrs.get("status", "todo"),
                "priority": attrs.get("priority", "medium"),
                "source_text": attrs.get("source_text"),
            })

    written = store.save_extracted_tasks(task_records)
    print("tasks_written:", written)

    # 3) Read calendar range
    tasks = store.get_tasks_by_date_range(
        start_date="2026-01-01",
        end_date="2026-12-31",
        workspace_id="ws-9",
        user_id="user-42"
    )
    print("tasks_in_range:", len(tasks))

    # 4) Graph + intelligence
    kg = KnowledgeGraph()
    kg.add_objects(result.objects)
    kg.add_links(result.links)

    intel = IntelligenceLayer(graph=kg)
    summary = intel.summarize_tasks(tasks, workspace_id="ws-9", user_id="user-42")
    print("summary_total:", summary["total_tasks"])
    print("summary_overdue:", summary["overdue"])
    print("summary_due_today:", summary["due_today"])

    # 5) Task search
    engine = HybridSearchEngine()
    hits = engine.search_tasks_nl(
        tasks,
        "due this week",
        workspace_id="ws-9",
        user_id="user-42"
    )
    print("search_hits_due_this_week:", len(hits))

    # Show 1 sample task
    if tasks:
        print("sample_task:", tasks[0])

if __name__ == "__main__":
    main()
