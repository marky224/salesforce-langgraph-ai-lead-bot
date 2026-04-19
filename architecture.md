# Architecture

Detailed architecture diagrams for the AI Sales Lead Bot.

---

## System Architecture

```mermaid
graph TB
    subgraph "Visitor's Browser"
        V[Website Visitor]
    end

    subgraph "GitHub Pages — markandrewmarquez.com"
        GP[index.html<br/>+ embed snippet]
    end

    subgraph "Azure Static Web Apps — Private Repo"
        WJS[widget.js]
        WCSS[chat-widget.css]
        NLUX[nlux Chat UI]
    end

    subgraph "Azure Container Apps — Docker"
        API[FastAPI Server<br/>POST /chat/stream<br/>POST /chat/init<br/>GET /health]
        LG[LangGraph Engine<br/>StateGraph + MemorySaver]
        NODES[Graph Nodes<br/>greeting → discovery →<br/>qualification → lead_capture →<br/>confirmation → scoring → SF]
    end

    subgraph "LLM Provider — Swappable"
        LLM[Claude / GPT-4o /<br/>Llama / Grok]
    end

    subgraph "Salesforce Developer Edition"
        LEAD[Lead Record<br/>+ Custom Fields]
        TASK[Task Record<br/>Chat Transcript]
        FLOW[Record-Triggered Flow<br/>LeadSource = Web Chat]
        AGENT[Agentforce Agent<br/>Lead Qualification Follow-Up]
        ACTIONS[Follow-up Tasks<br/>Email Draft<br/>Opportunity]
    end

    V --> GP
    GP -->|"<script src='...widget.js'>"| WJS
    WJS --> NLUX
    NLUX -->|"POST /chat/stream (SSE)"| API
    API --> LG
    LG --> NODES
    NODES -->|"ainvoke()"| LLM
    NODES -->|"simple_salesforce"| LEAD
    NODES -->|"simple_salesforce"| TASK
    LEAD -->|"After Insert"| FLOW
    FLOW -->|"Invoke"| AGENT
    AGENT --> ACTIONS
    API -->|"SSE tokens"| NLUX
```

---

## LangGraph Conversation Flow

```mermaid
stateDiagram-v2
    [*] --> entry_point

    entry_point --> greeting: No human messages
    entry_point --> extraction: Has human messages

    greeting --> WAIT: Reply sent

    extraction --> discovery: Stage = GREETING (fast-path)
    extraction --> router: Other stages
    extraction --> scoring: Stage = CONFIRMATION + confirmed
    extraction --> error: Error present

    router --> discovery: Needs discovery
    router --> qualification: Needs qualification
    router --> objection_handling: Objection detected
    router --> lead_capture: Ready for contact info
    router --> confirmation: Contact info complete
    router --> scoring: Stage = COMPLETE

    discovery --> WAIT
    qualification --> WAIT
    objection_handling --> WAIT
    lead_capture --> WAIT
    confirmation --> WAIT

    WAIT --> entry_point: Next human message

    scoring --> salesforce
    salesforce --> [*]: Success
    salesforce --> error: API failure
    error --> WAIT: Fallback message sent
```

---

## Data Model

```mermaid
erDiagram
    LEAD ||--o{ TASK : "WhoId"
    LEAD ||--o| OPPORTUNITY : "ConvertedOpportunityId"

    LEAD {
        string FirstName
        string LastName
        string Email
        string Company
        string Phone
        string Title
        string LeadSource "Web Chat"
        string Status "New → Working"
        string Description "Transcript summary"
        number Lead_Score__c "0-100"
        string Budget_Range__c "Picklist"
        string Timeline__c "Picklist"
        string Pain_Points__c "Long Text"
        string Company_Size__c "Picklist"
        string Chat_Transcript_ID__c "Task ID"
    }

    TASK {
        string Subject "AI Chat Transcript - date"
        string Description "Full transcript"
        string Status "Completed"
        string Priority "Normal"
        date ActivityDate "Today"
        reference WhoId "Lead ID"
    }

    OPPORTUNITY {
        string Name "Company - Web Chat Inquiry"
        string StageName "Qualification"
        currency Amount "Midpoint of budget"
        date CloseDate "Based on timeline"
        string LeadSource "Web Chat"
    }
```

---

## Graph State Schema

```mermaid
classDiagram
    class GraphState {
        +list~AnyMessage~ messages
        +ConversationStage stage
        +dict lead_data
        +dict qualification_data
        +int lead_score
        +dict lead_score_breakdown
        +list~str~ objections
        +str transcript_summary
        +str salesforce_lead_id
        +str salesforce_task_id
        +int retry_count
        +str error
    }

    class ConversationStage {
        <<enumeration>>
        GREETING
        DISCOVERY
        QUALIFICATION
        OBJECTION_HANDLING
        LEAD_CAPTURE
        CONFIRMATION
        COMPLETE
    }

    class LeadData {
        +str first_name
        +str last_name
        +str email
        +str company
        +str phone
        +str title
        +is_complete() bool
        +to_salesforce_payload() dict
    }

    class QualificationData {
        +BudgetRange budget_range
        +Timeline timeline
        +CompanySize company_size
        +list~str~ pain_points
        +bool decision_maker
        +str current_solution
        +list~str~ goals
        +to_salesforce_fields() dict
    }

    GraphState --> ConversationStage
    GraphState ..> LeadData : "lead_data dict"
    GraphState ..> QualificationData : "qualification_data dict"
```

---

## Scoring Rubric

```mermaid
pie title Lead Score Breakdown (100 points max)
    "Budget (25)" : 25
    "Timeline (20)" : 20
    "Company Size (15)" : 15
    "Decision Maker (15)" : 15
    "Pain Points (15)" : 15
    "Contact Completeness (10)" : 10
```

| Dimension | Points | High Value | Low Value |
|---|---|---|---|
| Budget | 0-25 | $100K+ = 25 pts | Unknown = 0 pts |
| Timeline | 0-20 | Immediate = 20 pts | Just exploring = 2 pts |
| Company Size | 0-15 | 1000+ = 15 pts | 1-10 = 4 pts |
| Decision Maker | 0-15 | Yes = 15 pts | Unknown = 0 pts |
| Pain Points | 0-15 | 3+ = 15 pts | 0 = 0 pts |
| Contact Completeness | 0-10 | All fields = 10 pts | No email = 0 pts |

| Priority | Score Range | Follow-up Actions |
|---|---|---|
| **High** | 70-100 | 3 tasks + email + opportunity (if budget ≥ $50K) |
| **Medium** | 40-69 | 2 tasks + email |
| **Low** | 0-39 | 1 nurture task |
