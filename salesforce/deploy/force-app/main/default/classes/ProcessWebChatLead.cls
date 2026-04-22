public with sharing class ProcessWebChatLead {

    public class ProcessRequest {
        @InvocableVariable(required=true description='The 18-character Salesforce Lead ID')
        public Id leadId;

        @InvocableVariable(required=true description='The numeric lead score 0-100')
        public Decimal leadScore;
    }

    public class ProcessResult {
        @InvocableVariable(description='Whether all steps completed successfully')
        public Boolean success;

        @InvocableVariable(description='Rating assigned: Hot, Warm, or Cold')
        public String rating;

        @InvocableVariable(description='Number of follow-up tasks created')
        public Integer taskCount;

        @InvocableVariable(description='Whether a follow-up email draft was created')
        public Boolean emailDrafted;

        @InvocableVariable(description='Whether an Opportunity was created')
        public Boolean opportunityCreated;

        @InvocableVariable(description='Opportunity ID if created')
        public String opportunityId;

        @InvocableVariable(description='Summary of all actions taken')
        public String summary;

        @InvocableVariable(description='Error message if any step failed')
        public String errorMessage;
    }

    @InvocableMethod(
        label='Process Web Chat Lead'
        description='Executes all follow-up steps for a web chat lead in one call: updates lead fields, creates tasks, drafts email, and optionally creates an opportunity.'
        category='Lead Management'
    )
    public static List<ProcessResult> processLead(List<ProcessRequest> requests) {
        List<ProcessResult> results = new List<ProcessResult>();

        for (ProcessRequest req : requests) {
            ProcessResult result = new ProcessResult();
            result.success = true;
            result.emailDrafted = false;
            result.opportunityCreated = false;
            result.taskCount = 0;

            List<String> summaryParts = new List<String>();

            try {
                Lead leadRecord = [
                    SELECT Id, FirstName, LastName, Company, Email,
                           Lead_Score__c, Budget_Range__c, Timeline__c,
                           Pain_Points__c, Company_Size__c, Description,
                           OwnerId
                    FROM Lead WHERE Id = :req.leadId LIMIT 1
                ];

                // STEP 1: Update Lead
                String rating;
                String status = null;
                if (req.leadScore >= 70) {
                    rating = 'Hot';
                    status = 'Working - Contacted';
                } else if (req.leadScore >= 40) {
                    rating = 'Warm';
                } else {
                    rating = 'Cold';
                }

                String analysisNote = '\n\n--- Agent Analysis ---\n'
                    + 'Lead score: ' + String.valueOf(req.leadScore.intValue())
                    + '. Rating: ' + rating + '. ';
                if (rating == 'Hot') {
                    analysisNote += 'High-priority lead with strong buying signals. Immediate follow-up recommended.';
                } else if (rating == 'Warm') {
                    analysisNote += 'Moderate interest detected. Nurture with relevant content.';
                } else {
                    analysisNote += 'Low engagement score. Add to long-term nurture campaign.';
                }

                leadRecord.Rating = rating;
                if (status != null) { leadRecord.Status = status; }
                leadRecord.Description = (leadRecord.Description != null ? leadRecord.Description : '') + analysisNote;
                update leadRecord;
                result.rating = rating;
                summaryParts.add('Rating set to ' + rating);
                if (status != null) { summaryParts.add('Status updated to ' + status); }

                // STEP 2: Create Follow-Up Tasks
                List<Task> tasks = new List<Task>();
                Integer scoreInt = req.leadScore.intValue();
                String leadName = '';
                if (leadRecord.FirstName != null) {
                    leadName = leadRecord.FirstName;
                    if (leadRecord.LastName != null) { leadName += ' ' + leadRecord.LastName; }
                } else if (leadRecord.LastName != null) {
                    leadName = leadRecord.LastName;
                }
                String companyName = leadRecord.Company != null ? leadRecord.Company : 'Unknown Company';

                Task callTask = new Task();
                callTask.WhoId = req.leadId;
                callTask.OwnerId = leadRecord.OwnerId;
                callTask.Subject = 'Call ' + leadName + ' — Discuss Requirements';
                callTask.Description = 'Follow up on web chat conversation. Lead score: ' + scoreInt
                    + '. Pain points: ' + (leadRecord.Pain_Points__c != null ? leadRecord.Pain_Points__c : 'Not captured');
                callTask.Priority = scoreInt >= 70 ? 'High' : 'Normal';
                callTask.Status = 'Not Started';
                callTask.ActivityDate = Date.today().addDays(1);
                tasks.add(callTask);

                if (scoreInt >= 40) {
                    Task caseStudyTask = new Task();
                    caseStudyTask.WhoId = req.leadId;
                    caseStudyTask.OwnerId = leadRecord.OwnerId;
                    caseStudyTask.Subject = 'Send Case Study to ' + leadName;
                    caseStudyTask.Description = 'Company: ' + companyName
                        + '. Size: ' + (leadRecord.Company_Size__c != null ? leadRecord.Company_Size__c : 'Unknown')
                        + '. Budget: ' + (leadRecord.Budget_Range__c != null ? leadRecord.Budget_Range__c : 'Unknown');
                    caseStudyTask.Priority = 'Normal';
                    caseStudyTask.Status = 'Not Started';
                    caseStudyTask.ActivityDate = Date.today().addDays(2);
                    tasks.add(caseStudyTask);
                }

                if (scoreInt >= 70) {
                    Task demoTask = new Task();
                    demoTask.WhoId = req.leadId;
                    demoTask.OwnerId = leadRecord.OwnerId;
                    demoTask.Subject = 'Schedule Demo for ' + leadName + ' at ' + companyName;
                    demoTask.Description = 'High-priority lead. Timeline: '
                        + (leadRecord.Timeline__c != null ? leadRecord.Timeline__c : 'Unknown')
                        + '. Budget: ' + (leadRecord.Budget_Range__c != null ? leadRecord.Budget_Range__c : 'Unknown');
                    demoTask.Priority = 'High';
                    demoTask.Status = 'Not Started';
                    demoTask.ActivityDate = Date.today().addDays(3);
                    tasks.add(demoTask);
                }

                insert tasks;
                result.taskCount = tasks.size();
                summaryParts.add(tasks.size() + ' follow-up tasks created');

                // STEP 3: Draft Follow-Up Email
                String firstName = leadRecord.FirstName != null ? leadRecord.FirstName : 'there';
                String painPoints = leadRecord.Pain_Points__c != null ? leadRecord.Pain_Points__c : 'your business challenges';
                String timeline = leadRecord.Timeline__c != null ? leadRecord.Timeline__c : 'your timeline';
                String budgetRange = leadRecord.Budget_Range__c != null ? leadRecord.Budget_Range__c : '';

                String cta;
                if (scoreInt >= 70) { cta = 'schedule a brief call'; }
                else if (scoreInt >= 40) { cta = 'review a relevant case study'; }
                else { cta = 'learn more about our solutions'; }

                String emailBody = 'Hi ' + firstName + ',\n\n'
                    + 'Thank you for chatting with us today about ' + painPoints + '. '
                    + 'I understand how important it is to find the right solution, '
                    + 'especially with ' + timeline + ' in mind.\n\n'
                    + 'Based on our conversation, I believe we can help '
                    + companyName + ' address these challenges effectively. '
                    + 'I would love to ' + cta + ' to explore how our solutions '
                    + 'align with your needs.\n\nLooking forward to connecting!\n\n'
                    + 'Best regards,\nMark Marquez Consulting';

                Task emailTask = new Task();
                emailTask.WhoId = req.leadId;
                emailTask.OwnerId = leadRecord.OwnerId;
                emailTask.Subject = 'Follow-Up Email Draft — ' + leadName;
                emailTask.Description = 'Subject: Following up on our conversation, ' + firstName + '\n\n' + emailBody;
                emailTask.Priority = 'Normal';
                emailTask.Status = 'Completed';
                emailTask.ActivityDate = Date.today();
                emailTask.Type = 'Email';
                insert emailTask;
                result.emailDrafted = true;
                summaryParts.add('Follow-up email drafted');

                // STEP 4: Create Opportunity (Conditional)
                Boolean qualifiesForOpp = scoreInt >= 80
                    && budgetRange != null
                    && (budgetRange.contains('50K') || budgetRange.contains('100K'));

                if (qualifiesForOpp) {
                    Decimal amount = 75000;
                    if (budgetRange.contains('100K')) { amount = 150000; }

                    Integer monthsToAdd = 3;
                    if (leadRecord.Timeline__c != null) {
                        if (leadRecord.Timeline__c == 'Immediate') { monthsToAdd = 1; }
                        else if (leadRecord.Timeline__c == '1-3 months') { monthsToAdd = 2; }
                        else if (leadRecord.Timeline__c == '3-6 months') { monthsToAdd = 4; }
                        else if (leadRecord.Timeline__c == '6+ months') { monthsToAdd = 7; }
                    }

                    Opportunity opp = new Opportunity();
                    opp.Name = companyName + ' — Web Chat Lead';
                    opp.StageName = 'Qualification';
                    opp.CloseDate = Date.today().addMonths(monthsToAdd);
                    opp.Amount = amount;
                    opp.LeadSource = 'Web Chat';
                    opp.Description = 'Auto-created from web chat lead. Score: ' + scoreInt + '. Rating: ' + rating + '.';
                    insert opp;
                    result.opportunityCreated = true;
                    result.opportunityId = opp.Id;
                    summaryParts.add('Opportunity created: ' + opp.Id);
                } else {
                    summaryParts.add('Opportunity not created — ' + (scoreInt < 80 ? 'score below 80' : 'budget below 50K'));
                }

                result.summary = String.join(summaryParts, '. ') + '.';

            } catch (Exception e) {
                result.success = false;
                result.errorMessage = e.getMessage();
                result.summary = 'Error: ' + e.getMessage();
            }

            results.add(result);
        }
        return results;
    }
}