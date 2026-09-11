"""
End-to-End Agentic Workflow Demonstration Script for CivicFlow AI (Phase 3).

Demonstrates complete complaint flow:
Citizen Complaint -> UnderstandingAgent -> ClassificationAgent -> TFIDF / Embedding / LLM Fallback -> Trace & ClassificationOutput
"""

import json
import sys
from pathlib import Path

# Ensure root workspace in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.agents.understanding_agent import UnderstandingAgent
from ai.agents.classification_agent import ClassificationAgent


DEMO_COMPLAINTS = [
    # 1. Clean TF-IDF Early Accept Case
    {
        "description": "Standard Garbage Complaint (TF-IDF High Confidence Early Accept)",
        "text": "Garbage vehicle has not arrived for three days in society. Trash is overflowing near main gate."
    },
    # 2. Marathi Transliterated Complaint (TF-IDF Match)
    {
        "description": "Marathi Streetlight Outage (Hinglish Transliterated)",
        "text": "Rastyavarchi street light maagchya aathavdya pasun chalat nahiye, full andhar aahe."
    },
    # 3. Contextual Complaint (Escalation Case)
    {
        "description": "Unusual Contextual Pothole Complaint (Triggers Tool Escalation)",
        "text": "For the past several mornings our society residents bike wheels are getting damaged on the flyover bend craters."
    },
    # 4. Critical Safety Hazard
    {
        "description": "Critical Electrical & Fire Hazard (Safety Signal Flagged)",
        "text": "Live electrical wire hanging low near children park gate posing severe shock hazard."
    },
    # 5. Unmapped Property Tax Complaint
    {
        "description": "Property Tax Assessment Error (Unmapped Dept)",
        "text": "Property tax online bill double calculation error. Received incorrect penalty notice."
    },
    # 6. Non-Municipal Out-of-Scope Complaint
    {
        "description": "Non-Municipal Query (Out-Of-Scope Filtered)",
        "text": "My mobile phone battery drains very fast and cricket match score."
    }
]


def run_demo():
    understanding_agent = UnderstandingAgent()
    classification_agent = ClassificationAgent()

    print("==================================================")
    print("CIVICFLOW AI - END-TO-END AGENTIC DEMONSTRATION")
    print("==================================================")

    for idx, case in enumerate(DEMO_COMPLAINTS, 1):
        print(f"\n[{idx}] Case Description: {case['description']}")
        print(f"    Raw Complaint Text: \"{case['text']}\"")

        # Step 1: Understanding Agent
        understanding = understanding_agent.understand(case["text"])
        print(f"    1. Understanding Agent Output:")
        print(f"       - Detected Language: {understanding.language.code} (Confidence: {understanding.language.confidence:.1%})")
        if understanding.safety_signals:
            print(f"       - [!] Safety Hazard Alert: {understanding.safety_signals[0].type} ({understanding.safety_signals[0].severity})")
        if understanding.missing_information:
            print(f"       - Missing Info: {understanding.missing_information}")

        # Step 2: Classification Agent
        classification = classification_agent.classify(understanding, debug=True)
        print(f"    2. Classification Agent Output:")
        print(f"       - Status: {classification.status}")
        print(f"       - Method Selected: {classification.method}")
        print(f"       - Domain:   {classification.semantic_classification.domain_code} ({classification.semantic_classification.domain_name})")
        print(f"       - Category: {classification.semantic_classification.category_code} ({classification.semantic_classification.category_name})")
        print(f"       - Overall Confidence: {classification.overall_confidence:.1%}")
        print(f"       - Admin Department:   {classification.administrative_mapping.type} (Dept: {classification.administrative_mapping.department_code})")

        # Step 3: Explainable Agent Trace
        trace = classification.secondary_signals[0].get("trace", []) if classification.secondary_signals else []
        print(f"    3. Explainable Agent Execution Trace:")
        for step in trace:
            dec = step.get("decision")
            tool_name = step.get("tool")
            conf = step.get("confidence", 0.0)
            reason = step.get("reason", "")
            print(f"       -> [{tool_name:12s}] Decision: {dec:8s} | Confidence: {conf:.1%} {f'| Reason: {reason}' if reason else ''}")

    print("\n==================================================")
    print("DEMONSTRATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    run_demo()
