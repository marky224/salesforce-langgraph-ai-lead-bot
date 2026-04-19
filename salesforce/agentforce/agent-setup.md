# Agentforce Agent Setup — Step-by-Step

How to create and configure the **Lead Qualification Follow-Up Agent**
in Salesforce Setup.

---

## Prerequisites

- Salesforce Developer Edition with **Agentforce** enabled
- Einstein features activated in your org
- The custom Lead fields from `custom-fields.md` already created
- The Record-Triggered Flow from `flows/flow-setup.md` ready (but not yet
  activated — activate it after the agent is configured)

> **Note:** Agentforce availability varies by Salesforce edition and release.
> If Agentforce is not available in your org, use the Flow-only fallback
> described in `flows/Lead_Created_Flow.md`.

---

## Step 1: Enable Agentforce

1. **Setup** → search "Einstein Setup" → click **Einstein Setup**
2. Toggle **Turn on Einstein** to enabled (if not already)
3. **Setup** → search "Agents" → click **Agents** (under Einstein)
4. If prompted, accept the terms and enable Agentforce

---

## Step 2: Create the Agent

1. **Setup** → search "Agents" → click **Agents**
2. Click **New Agent**
3. Configure:

   | Field | Value |
   |---|---|
   | Agent Name | `Lead Qualification Follow-Up Agent` |
   | API Name | `Lead_Qualification_Follow_Up_Agent` |
   | Description | Processes newly created web chat leads, analyzes conversation transcripts, scores lead quality, creates follow-up tasks, drafts personalized emails, and creates opportunities for high-value leads. |
   | Agent Type | Autonomous (if available) or Standard |

4. Click **Save**

---

## Step 3: Add the Agent Instructions

1. Open the agent you just created
2. Find the **Instructions** field (also called "System Prompt" or "Agent Prompt")
3. Paste the complete instruction block from `agent-instructions.md`
   (everything between the triple backticks)
4. Click **Save**

---

## Step 4: Create Topic 1 — Lead Analysis

1. In the agent configuration, go to **Topics**
2. Click **New Topic**
3. Configure:

   | Field | Value |
   |---|---|
   | Topic Label | Lead Analysis |
   | Description | Analyze a newly created web chat Lead by reading the Lead record and associated conversation transcript. Identify buying signals, objections, pain points, and urgency level. |
   | Topic Instructions | Paste from `agent-topics.md` → Topic 1 → Instructions |

4. Add **Actions**:
   - Click **Add Action**
   - Search for and add record query actions for Lead and Task
   - If pre-built actions aren't available, you'll need to create
     **Apex Invocable Actions** (see Step 8 below)
5. Click **Save**

---

## Step 5: Create Topic 2 — Follow-Up Task Creation

1. Click **New Topic**
2. Configure:

   | Field | Value |
   |---|---|
   | Topic Label | Follow-Up Task Creation |
   | Description | Create prioritized follow-up Tasks assigned to the Lead owner based on lead analysis. |
   | Topic Instructions | Paste from `agent-topics.md` → Topic 2 → Instructions |

3. Add **Actions**:
   - Add a record create action for Task
4. Click **Save**

---

## Step 6: Create Topic 3 — Email Drafting

1. Click **New Topic**
2. Configure:

   | Field | Value |
   |---|---|
   | Topic Label | Email Drafting |
   | Description | Draft a personalized follow-up email and store as an Email-type Task. |
   | Topic Instructions | Paste from `agent-topics.md` → Topic 3 → Instructions |

3. Add **Actions**:
   - Add a record create action for Task (Type = Email)
4. Click **Save**

---

## Step 7: Create Topic 4 — Opportunity Creation

1. Click **New Topic**
2. Configure:

   | Field | Value |
   |---|---|
   | Topic Label | Opportunity Creation |
   | Description | Create an Opportunity for high-value leads meeting score and budget thresholds. |
   | Topic Instructions | Paste from `agent-topics.md` → Topic 4 → Instructions |

3. Add **Actions**:
   - Add record create action for Opportunity
   - Add record update action for Lead
4. Click **Save**

---

## Step 8: Create Apex Invocable Actions (If Needed)

If the agent's built-in record actions aren't sufficient, create Apex
classes with `@InvocableMethod` annotations that the agent can call.

### Example: Query Transcript Task

```apex
public class GetTranscriptTask {
    @InvocableMethod(
        label='Get Chat Transcript'
        description='Retrieves the most recent AI Chat Transcript Task for a Lead'
    )
    public static List<Task> getTranscript(List<Id> leadIds) {
        return [
            SELECT Id, Subject, Description, CreatedDate
            FROM Task
            WHERE WhoId IN :leadIds
              AND Subject LIKE 'AI Chat Transcript%'
            ORDER BY CreatedDate DESC
            LIMIT 1
        ];
    }
}
```

### Example: Create Follow-Up Task

```apex
public class CreateFollowUpTask {

    public class TaskInput {
        @InvocableVariable(required=true label='Lead ID')
        public Id leadId;

        @InvocableVariable(required=true label='Subject')
        public String subject;

        @InvocableVariable(label='Description')
        public String description;

        @InvocableVariable(required=true label='Priority')
        public String priority;

        @InvocableVariable(required=true label='Days Until Due')
        public Integer daysUntilDue;
    }

    @InvocableMethod(
        label='Create Follow-Up Task'
        description='Creates a follow-up Task linked to a Lead'
    )
    public static List<Id> createTask(List<TaskInput> inputs) {
        List<Task> tasks = new List<Task>();
        List<Lead> leads = [
            SELECT Id, OwnerId FROM Lead WHERE Id IN
                (SELECT leadId FROM TaskInput)
        ];
        Map<Id, Id> ownerMap = new Map<Id, Id>();
        for (Lead l : leads) ownerMap.put(l.Id, l.OwnerId);

        for (TaskInput input : inputs) {
            tasks.add(new Task(
                WhoId = input.leadId,
                OwnerId = ownerMap.get(input.leadId),
                Subject = input.subject,
                Description = input.description,
                Priority = input.priority,
                Status = 'Not Started',
                ActivityDate = Date.today().addDays(input.daysUntilDue)
            ));
        }
        insert tasks;

        List<Id> taskIds = new List<Id>();
        for (Task t : tasks) taskIds.add(t.Id);
        return taskIds;
    }
}
```

Deploy these classes via **Setup** → **Apex Classes** → **New**, or via
your preferred deployment tool (SFDX, VS Code with Salesforce Extension Pack).

---

## Step 9: Connect the Flow to the Agent

Update the Record-Triggered Flow to invoke this agent:

1. **Setup** → **Flows** → open `AI Chat Lead - Post-Creation Automation`
2. Replace the direct Task creation elements with an **Action** element
   that invokes the Agentforce agent
3. Pass the Lead ID and other fields as inputs
4. Save and activate the Flow

> If the Flow can't directly invoke the agent, use an Apex Invocable
> Action as the bridge — the Apex class calls the agent via the
> Agentforce API.

---

## Step 10: Test the Agent

### Manual Testing (Agent Builder)

1. Open the agent in Setup
2. Use the **Test** panel (right sidebar)
3. Provide a test prompt:
   ```
   Process Lead 00Q000000000001. The lead score is 85, budget is
   $50K-$100K, timeline is 1-3 months. Pain points: manual data entry,
   no CRM integration. Company: Acme Corp, Contact: Sarah Chen.
   ```
4. Verify the agent:
   - Creates 3 follow-up Tasks
   - Drafts a personalized email
   - Creates an Opportunity (score 85, budget $50K+)
   - Updates Lead status

### End-to-End Testing

1. Start the backend locally
2. Complete a full chat conversation through the widget
3. Check Salesforce:
   - Lead created with all custom fields populated
   - Transcript Task attached
   - Flow triggered → Agent invoked
   - Follow-up Tasks created with specific details from the conversation
   - Email draft Task exists
   - Opportunity created if thresholds met

---

## Step 11: Activate for Production

1. Verify all tests pass
2. Ensure the agent is set to **Active**
3. Ensure the Record-Triggered Flow is **Active**
4. Monitor the first few production leads:
   - **Setup** → **Agents** → agent detail → **Logs/History**
   - **Setup** → **Flows** → flow → **Run History**
5. Adjust agent instructions or topic parameters based on output quality

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| Agent doesn't fire | Check Flow is active, entry conditions match, and the Flow's agent action is configured |
| Tasks created without specific details | Refine the topic instructions — add more emphasis on referencing transcript content |
| Opportunity not created for high-score leads | Verify Budget_Range__c field value matches exactly ("$50K-$100K" not "$50K - $100K") |
| Agent produces generic emails | Add more examples to the Email Drafting topic instructions |
| "Insufficient access" errors | Check the agent's running user has CRUD on Lead, Task, and Opportunity |
| Agent available but won't show | Ensure Einstein is enabled and Agentforce feature is activated in your org |
