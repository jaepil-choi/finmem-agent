import logging
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.rag.strategies.finmem import FinMemRAG

logger = logging.getLogger(__name__)

def memory_update_node(state: GraphState) -> Dict[str, Any]:
    """
    Single node that applies all accumulated memory updates to the FAISS index.
    Prevents race conditions by running after all parallel reflection nodes.
    """
    updates = state.get("memory_updates", [])
    if not updates:
        return {}

    logger.info(f"Applying {len(updates)} memory updates to FAISS...")
    print(f"\n    [Memory Update] Aggregating updates for {len(updates)} citations...")

    strategy = FinMemRAG()
    indices_updated = set()

    # Aggregate feedback by document (source_id + page_content hash or equality)
    # This prevents multiple saves for the same document and correctly aggregates feedback
    aggregated_feedback = {}
    
    for up in updates:
        # Use a combination of source_id and content as a unique key for the document in the turn
        key = (up["tier"], up["source_id"], up["page_content"])
        if key not in aggregated_feedback:
            aggregated_feedback[key] = {
                "feedback": 0.0,
                "filename": up["filename"],
                "doc_id_sample": up["doc_id"]
            }
        aggregated_feedback[key]["feedback"] += up["feedback"]

    for (tier, source_id, content), info in aggregated_feedback.items():
        if tier not in strategy.indices:
            continue
            
        index = strategy.indices[tier]
        feedback = info["feedback"]
        doc_id = info["doc_id_sample"]
        
        target_found = False
        for internal_id, stored_doc in index.docstore._dict.items():
            if stored_doc.page_content == content and \
               stored_doc.metadata.get("source_id") == source_id:
                
                # Found the document! Apply FinMem logic
                ac = stored_doc.metadata.get("access_counter", 0)
                # Ensure importance is at least 40 if missing or 0
                importance = stored_doc.metadata.get("importance", 40.0)
                if importance == 0: importance = 40.0
                
                new_ac = ac + feedback
                # Formula: importance increases/decreases by feedback * factor
                # This is an incremental update
                new_importance = importance + (feedback * 5.0)
                new_importance = max(0.0, min(100.0, new_importance))
                
                stored_doc.metadata["access_counter"] = new_ac
                stored_doc.metadata["importance"] = float(new_importance)
                
                change_type = "INCREASE" if feedback > 0 else "DECREASE"
                print(f"      [{change_type}] Doc {doc_id} ({info['filename']}): Importance {importance:.1f} -> {new_importance:.1f} (AC: {ac} -> {new_ac})")
                
                logger.info(f"Updated doc {doc_id} in tier {tier}: AC {ac}->{new_ac}, Importance {importance}->{new_importance:.2f}")
                target_found = True
                indices_updated.add(tier)
                break
        
        if not target_found:
            logger.warning(f"Could not find document for update: {info['filename']}")

    # Persist changes
    for tier in indices_updated:
        tier_path = strategy.vector_db_root / tier
        strategy.indices[tier].save_local(str(tier_path))
        logger.info(f"Persisted updated FAISS index for tier: {tier}")

    return {"memory_updates": []} # Clear updates after applying
