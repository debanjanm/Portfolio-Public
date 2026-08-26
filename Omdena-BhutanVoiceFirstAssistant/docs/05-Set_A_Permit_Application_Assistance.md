# Content Set A — Permit Application Assistance

Version: Draft v2
Status: Pending Team 1 Validation
Prepared For:
- Team 2 OpenSearch Ingestion
- Aurora PostgreSQL Rules Integration
- Voice-AI Workflow Engine

Language: English
Primary Domain: Timber, Sand, Stone Permit Assistance

## 1. Purpose

This content set helps citizens understand timber, sand, and stone permit procedures in Bhutan.

The content is designed for:
- Voice-first AI interaction
- Rural-first usability
- Low digital literacy users
- Weak connectivity environments

The content includes:
- Eligibility guidance
- Required documents
- Permit categories
- Step-by-step process guidance
- Workflow rules
- Human escalation conditions
- Connectivity-aware interaction guidance

## 2. Metadata

- Content Set: A
- Services Covered:
  - Timber Permit
  - Sand Permit
  - Stone Permit
- Target Interaction Mode:
  - Voice
  - Mobile
  - Community Centre Support
- Supported Languages:
  - Dzongkha
  - English

### A1.1 What is a Timber Permit?

A timber permit allows eligible citizens to apply for timber for approved rural and personal use.

Examples include:
- New rural house construction
- House repair or renovation
- Livestock shelter
- Makeshift shack
- Firewood
- Fencing post
- Flag pole

### A1.2 Supported Timber Request Categories

The system should support the following timber request types:

1. New rural house construction
2. House repair / renovation / extension
3. Livestock shelter
4. Makeshift shack
5. Firewood permit
6. Fencing post permit
7. Flag pole permit

### A1.3 Eligibility Criteria for Rural Subsidized Timber

The applicant must:
- Be native of the area
- Be Head of Gung (village household)
- The land must be registered in the applicant’s name
- Timber must be for bona-fide rural use
- The land should be inherited and not purchased
- Applicant must not already have received subsidized timber for another rural house

Approved uses include:
- Rural house construction
- House repair
- Livestock shelter
- Fencing post
- Flag pole
- Pyre wood

### A1.4 Restricted Areas

Subsidized timber is generally not allowed:
- Inside Thromde (municipal) areas
- Within 2 km radius of Thromde boundaries

Exceptions:
- Special native-area conditions
[Team 1 to verify]

### A1.5 Required Documents

For new construction:
- Citizenship Identity Card (CID)
- Construction approval
- Land ownership proof
- Valid phone number

For repair or renovation:
[Team 1 to verify]

For fencing/firewood/flag pole:
[Team 1 to verify]


### A1.6 Step-by-Step Application Guidance

Step 1:
Identify timber permit category.

Step 2:
Check eligibility.

Step 3:
Prepare required documents.

Step 4:
Choose application mode:
- Online portal
- Community Centre
- Forestry office

Step 5:
Fill application form.

Step 6:
Upload or submit documents.

Step 7:
Track permit status.

### A1.7 Weak Connectivity Handling

If connectivity becomes weak:
- Save current workflow progress
- Switch to low-bandwidth audio mode
- Offer SMS summary
- Resume interaction after reconnect

The system should avoid forcing users to restart applications.

### A1.8 Human Escalation Conditions

The system should escalate to a human officer if:
- User repeatedly fails steps
- User becomes confused
- Eligibility situation is unclear
- User becomes emotionally frustrated
- Connectivity repeatedly fails

### A1.9 Voice Interaction Principles

The assistant should:
- Use short sentences
- Ask one question at a time
- Avoid bureaucratic language
- Confirm each important step
- Prioritize Dzongkha interaction

### A1.10 Common Citizen Pain Points

The system should help reduce:
- Repeated travel
- Missing documents
- Confusion about eligibility
- Dependence on officials/family
- Connectivity-related failures

{
  "content_set": "A",
  "service": "Timber Permit",
  "permit_categories": [
    "new_house",
    "repair",
    "livestock_shelter",
    "makeshift_shack",
    "firewood",
    "fencing_post",
    "flag_pole"
  ],
  "eligibility_rules": [
    "must_be_head_of_gung",
    "land_registered_in_name",
    "rural_use_only",
    "not_within_thromde_boundary",
    "no_recent_subsidized_timber"
  ],
  "required_documents": {
    "new_house": [
      "CID",
      "construction_approval",
      "land_ownership_proof",
      "phone_number"
    ]
  },
  "workflow_steps": [
    "identify_category",
    "check_eligibility",
    "document_validation",
    "application_submission",
    "status_tracking"
  ],
  "human_escalation_conditions": [
    "repeated_confusion",
    "multiple_failed_attempts",
    "complex_eligibility",
    "emotional_frustration"
  ],
  "connectivity_handling": {
    "save_progress": true,
    "audio_only_mode": true,
    "resume_previous_state": true,
    "sms_summary": true
  }
}

## 7. Conversation State Flow

Greeting
→ Language Selection
→ Intent Detection
→ Eligibility Guidance
→ Document Guidance
→ Application Assistance
→ Connectivity Check
→ Submission
→ Human Escalation (if required)
→ End

## 8. Source References

- Online Forestry Services Portal
- Department of Forests and Park Services
- Timber Permit Workflow Document from Team 1
- Department of Geology and Mines