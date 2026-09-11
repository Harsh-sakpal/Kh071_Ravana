# CivicFlow AI — Person-2 AI Microservice Subsystem

**Production-Ready AI Agent Suite for Municipal Complaint Routing, Continuous DIPEX Prioritization, SLA Breach Risk Monitoring, and Automated Escalation.**

---

## 🏛️ Implemented Agents & Endpoints Overview

| Agent Name | Database `ai_agent_type` | Primary Role & Intelligence | REST Endpoint |
| :--- | :--- | :--- | :--- |
| **Routing Agent** | `ROUTING_AGENT` | Maps raw complaint text to 15 AI Domains, 225 subcategories, database department codes (`SAN`, `RDW`, `WTR`, `ELE`, `PRK`), and seeded `category_code`s. | `POST /api/v1/ai/routing` |
| **DIPEX Priority Agent** | `PRIORITY_AGENT` | Computes continuous DIPEX priority score ($0.0-1.0$) across 10 feature groups, triggers safety overrides for life hazards, and sets target SLA resolution hours (`6h`, `24h`, `48h`, `120h`). | `POST /api/v1/ai/priority` |
| **SLA Monitoring Agent** | `SLA_MONITORING_AGENT` | 6-factor breach probability engine ($0.0-1.0$), category-specific SLA duration lookup, 25% location discounts (schools/hospitals), and multi-tier status badges. | `POST /api/v1/ai/sla-monitor` |
| **Escalation Agent** | `ESCALATION_AGENT` | 5-level escalation matrix (`TIER_0` to `TIER_4`), automated officer workload rebalancing, SLA overtime penalty scoring, and SMS/Email alert draft generation. | `POST /api/v1/ai/escalate` |

---

## 🚀 Quickstart & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated test suite (50 tests)
pytest tests/ -v

# 3. Start FastAPI microservice
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger Documentation: `http://localhost:8000/docs`

---

## 📡 API Contract Reference

### 1. Routing Agent: `POST /api/v1/ai/routing`
**Request:**
```json
{
  "complaint_text": "Live sparking electrical wire hanging over pedestrian walkway outside Sector 4 Primary School."
}
```
**Response:**
```json
{
  "status": "SUCCESS",
  "domain": "DOM-ELE",
  "subcategory": "EXPOSED_ELECTRICAL_WIRE",
  "department_code": "ELE",
  "category_code": "ELE-LGT",
  "confidence": 0.95,
  "reason": "Routed to Electricity department (ELE) based on keywords: wire, electrical, sparking."
}
```

---

### 2. Priority Agent: `POST /api/v1/ai/priority`
**Request:**
```json
{
  "complaint_text": "Live sparking electrical wire hanging over walkway.",
  "domain": "DOM-ELE",
  "subcategory": "EXPOSED_ELECTRICAL_WIRE",
  "category_code": "ELE-LGT",
  "location_context": "Near Primary School Entrance",
  "affected_count": 80,
  "duration_days": 1.0
}
```
**Response:**
```json
{
  "status": "SUCCESS",
  "priority": "CRITICAL",
  "priority_score": 0.95,
  "sla_target_hours": 6,
  "confidence": 0.95,
  "override_applied": true,
  "override_reason": "CRITICAL SAFETY OVERRIDE TRIGGERED: Exposed electrical wire detected near school.",
  "factors": {
    "safety_hazard": 1.0,
    "vulnerable_population": 0.85,
    "affected_citizens": 0.65,
    "urgency": 0.9
  },
  "risk_factors": ["VULNERABLE_POPULATION_SCHOOL", "HIGH_SAFETY_HAZARD"]
}
```

---

### 3. SLA Monitoring Agent: `POST /api/v1/ai/sla-monitor`
**Request:**
```json
{
  "ticket_id": "TCK-2026-0042",
  "priority": "CRITICAL",
  "sla_target_hours": 6,
  "category_code": "ELE-LGT",
  "subcategory": "EXPOSED_ELECTRICAL_WIRE",
  "is_sensitive_location": true,
  "officer_workload_count": 3,
  "department_queue_count": 5,
  "created_at": "2026-09-11T00:00:00Z"
}
```
**Response:**
```json
{
  "status": "SUCCESS",
  "sla_status": "WITHIN_SLA",
  "sla_tier": "ON_TRACK",
  "elapsed_hours": 1.0,
  "remaining_hours": 3.5,
  "sla_target_hours": 4,
  "sla_ratio": 0.25,
  "breach_probability": 0.18,
  "sla_factors": {
    "time_elapsed": 0.25,
    "officer_workload": 0.4,
    "department_queue": 0.1,
    "complexity": 0.85,
    "location_urgency": 0.85,
    "historical_breach": 0.15
  },
  "recommendation_action": "MAINTAIN_NORMAL_MONITORING"
}
```

---

### 4. Escalation Agent: `POST /api/v1/ai/escalate`
**Request:**
```json
{
  "ticket_id": "TCK-2026-0042",
  "department_code": "ELE",
  "priority": "CRITICAL",
  "sla_status": "BREACHED",
  "sla_ratio": 1.15,
  "is_breached": true,
  "is_safety_override": true,
  "affected_count": 80,
  "current_officer_id": "OFF-ASSIGNED-01"
}
```
**Response:**
```json
{
  "status": "SUCCESS",
  "ticket_id": "TCK-2026-0042",
  "escalation_tier": "TIER_4_EMERGENCY_OVERRIDE",
  "action_required": "EMERGENCY_DISPATCH_TRIGGERED",
  "target_role": "EMERGENCY_RESPONSE_UNIT",
  "target_contact": "EMERGENCY-DISPATCH-ELE | Hotline 911-CIVIC",
  "recommended_officer_id": "OFF-ELE-02",
  "recommended_officer_name": "Electrician Robert Chen",
  "penalty_score": 195.0,
  "notification_draft": "[🚨 EMERGENCY DISPATCH ALERT]\nTicket ID: TCK-2026-0042\nDepartment: ELE | Priority: CRITICAL\nDIRECTIVE: Deploy emergency response unit immediately. Cross-department coordination activated.",
  "audit_trace": [
    "Initiated Escalation Engine evaluation for ticket 'TCK-2026-0042'.",
    "CRITICAL TRIGGER: Emergency breach condition met (Safety Override). Tier 4 active.",
    "Reassignment Engine selected 'Electrician Robert Chen' (OFF-ELE-02) with 1 active tickets."
  ]
}
```

---

## 📂 Package Architecture

```
civicflow-ai-person2/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── routing_agent.py          # 15 Domains / 225 Subcategories Rule Engine
│   │   ├── priority_agent.py         # DIPEX Continuous Scoring & Safety Override
│   │   ├── sla_monitoring_agent.py   # 6-Factor Breach Probability & Location Discounts
│   │   └── escalation_agent.py       # 5-Tier Escalation & Reassignment Engine
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routing.py                # POST /api/v1/ai/routing
│   │   ├── priority.py               # POST /api/v1/ai/priority
│   │   ├── sla_monitoring.py         # POST /api/v1/ai/sla-monitor
│   │   └── escalation.py             # POST /api/v1/ai/escalate
│   ├── config/
│   │   ├── __init__.py
│   │   ├── routing_config.py         # Taxonomies & Keywords
│   │   ├── priority_config.py        # DIPEX Weights & Safety Rules
│   │   ├── sla_config.py             # Category Durations & Tier Thresholds
│   │   └── escalation_config.py      # Escalation Matrices & Department Heads
│   ├── models/
│   │   ├── __init__.py
│   │   ├── routing.py                # Pydantic Schemas
│   │   ├── priority.py               # Pydantic Schemas
│   │   ├── sla_monitoring.py         # Pydantic Schemas
│   │   └── escalation.py             # Pydantic Schemas
│   ├── __init__.py
│   └── main.py                       # FastAPI Application Entrypoint
├── tests/
│   ├── __init__.py
│   ├── test_routing_agent.py         # 13 Tests
│   ├── test_priority_agent.py        # 19 Tests
│   ├── test_sla_monitoring_agent.py  # 10 Tests
│   └── test_escalation_agent.py      # 8 Tests
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

All **50 unit tests** can be run with:
```bash
pytest tests/ -v
```
Output:
```
======================= 50 passed, 2 warnings in 0.77s ========================
```

