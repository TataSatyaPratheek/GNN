# Process and Activities Being Mined

## The Core Process: Domestic Financial Declarations

The BPI2020 Domestic Declarations dataset captures the **expense reimbursement and financial declaration process** within an organization. This process allows employees to submit financial declarations for approval and subsequent payment. The process mining analysis reveals the complete lifecycle of these declarations from initial submission to final payment.

## Process Scope

The process scope encompasses:

1. **Declaration Initiation**: Employee submission of financial declarations
2. **Multi-level Review**: Various approval stages by different organizational roles
3. **Payment Processing**: System-based payment request and execution
4. **Exception Handling**: Rejection and resubmission workflows

The process does not appear to include pre-declaration activities (e.g., expense incurred) or post-payment activities (e.g., accounting reconciliation).

## Activities in the Process

The dataset contains **17 distinct activities** across the declaration lifecycle. These can be categorized as follows:

### 1. Submission Activities

- **Declaration SUBMITTED by EMPLOYEE**
  - Frequency: 11,531 occurrences (20.4% of all activities)
  - Performed by: EMPLOYEE role
  - Function: Initiates the declaration process
  - Input: Declaration details, amount, and potential attachments
  - Output: Declaration submitted for review
  
- **Declaration REJECTED by EMPLOYEE**
  - Frequency: 1,365 occurrences (2.4%)
  - Performed by: EMPLOYEE role
  - Function: Employee withdraws or rejects their own declaration
  - Input: Previously submitted declaration
  - Output: Declaration process terminated or restarted

### 2. Administrative Review Activities

- **Declaration APPROVED by ADMINISTRATION**
  - Frequency: 8,202 occurrences (14.5%)
  - Performed by: ADMINISTRATION role
  - Function: Initial verification of declaration completeness and validity
  - Input: Submitted declaration
  - Output: Declaration approved for next level review
  
- **Declaration REJECTED by ADMINISTRATION**
  - Frequency: 952 occurrences (1.7%)
  - Performed by: ADMINISTRATION role
  - Function: Return declaration to employee due to issues
  - Input: Submitted declaration with issues
  - Output: Declaration returned for correction

### 3. Management Approval Activities

- **Declaration APPROVED by SUPERVISOR**
  - Frequency: Unknown (combined with FINAL_APPROVED)
  - Performed by: SUPERVISOR role
  - Function: Line manager approval of the declaration
  - Input: Admin-approved declaration
  - Output: Declaration approved by supervisor
  
- **Declaration FINAL_APPROVED by SUPERVISOR**
  - Frequency: 10,131 occurrences (17.9%)
  - Performed by: SUPERVISOR role
  - Function: Final approval by line manager
  - Input: Previously approved declaration
  - Output: Declaration fully approved for payment

- **Declaration REJECTED by SUPERVISOR**
  - Frequency: 293 occurrences (0.5%)
  - Performed by: SUPERVISOR role
  - Function: Rejection of the declaration by line manager
  - Input: Admin-approved declaration
  - Output: Declaration returned for correction
  
- **Declaration APPROVED by BUDGET OWNER**
  - Frequency: 2,820 occurrences (5.0%)
  - Performed by: BUDGET OWNER role
  - Function: Approval based on budget responsibility
  - Input: Declaration requiring budget owner approval
  - Output: Budget-approved declaration
  
- **Declaration REJECTED by BUDGET OWNER**
  - Frequency: 97 occurrences (0.2%)
  - Performed by: BUDGET OWNER role
  - Function: Rejection based on budget considerations
  - Input: Declaration not meeting budget criteria
  - Output: Declaration returned for correction
  
- **Declaration APPROVED by PRE_APPROVER**
  - Frequency: 685 occurrences (1.2%)
  - Performed by: PRE_APPROVER role
  - Function: Preliminary approval for special cases
  - Input: Declarations requiring preliminary review
  - Output: Pre-approved declaration for further approval

- **Declaration REJECTED by PRE_APPROVER**
  - Frequency: 89 occurrences (0.2%)
  - Performed by: PRE_APPROVER role
  - Function: Preliminary rejection
  - Input: Declaration not meeting criteria
  - Output: Declaration returned for correction

### 4. Payment Processing Activities

- **Request Payment**
  - Frequency: 10,040 occurrences (17.8%)
  - Performed by: SYSTEM (automated)
  - Function: System-generated request for payment
  - Input: Fully approved declaration
  - Output: Payment request created
  
- **Payment Handled**
  - Frequency: 10,044 occurrences (17.8%)
  - Performed by: SYSTEM (automated)
  - Function: Confirmation of payment execution
  - Input: Payment request
  - Output: Completed payment, closed declaration

### 5. Other Activities

The dataset contains a few other less frequent activities related to:
- Special approvals
- Director involvement
- Missing information handling
- Delegation cases

## Activity Relationships

The activities form a structured process with defined sequences:

1. **Linear Path**: The standard process follows a linear path from submission through approvals to payment
2. **Conditional Branches**: Different approval paths based on declaration characteristics
3. **Loops**: Rejection activities create loops back to earlier stages for correction
4. **Parallel Paths**: Some declarations may require multiple types of approvals

## Activity Timing Patterns

Analysis of timestamps reveals:

1. **Working Hours Concentration**: Most human activities occur during standard working hours (9 AM - 5 PM)
2. **System Activity Patterns**: Payment activities often occur in batches at specific times
3. **Waiting Times**: Significant waiting times between certain activities, particularly:
   - After submission (average 16.3 hours)
   - After administrative approval (average 9.2 hours)
   - After budget owner approval (average 22.5 hours)
   - Before payment handling (average 48.1 hours)

## Activities by Resource Type

The activities in the dataset are performed by two resource types:

1. **STAFF MEMBER**: Handles all human approval and rejection activities
2. **SYSTEM**: Handles automated payment request and payment handling activities

## Process Mining Focus Areas

When mining this process, the analysis focuses on:

1. **Process Flow Discovery**: Understanding the actual sequences of activities 
2. **Variant Analysis**: Identifying common and exceptional paths through the process
3. **Bottleneck Detection**: Finding activities with significant waiting times
4. **Conformance Checking**: Comparing actual process execution with expected flows
5. **Organizational Perspective**: Analyzing how different roles interact within the process
6. **Performance Analysis**: Measuring throughput times, waiting times, and processing times

## Process Mining Challenges

The mining of this process presents several challenges:

1. **Concept Drift**: Potential changes in the process over the 2.5-year period
2. **Incomplete Traces**: Some cases may be truncated at dataset boundaries
3. **Activity Granularity**: Some activities may encapsulate multiple sub-activities not captured
4. **Missing Context**: External factors influencing process variations are not recorded
5. **Resource Anonymization**: Generic resource labels limit resource-based analysis

## Conclusion

The BPI2020 Domestic Declarations dataset allows for comprehensive process mining of a financial declaration approval process. The activities captured represent a complete view of the declaration lifecycle from submission to payment, with various approval paths and exception handling scenarios. The process exhibits characteristics of a well-structured administrative workflow with defined roles, clear activity sequences, and automated components.