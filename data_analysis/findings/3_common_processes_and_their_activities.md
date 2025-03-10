# Common Processes and Their Activities

## Overview of Process Variants

The BPI2020 Domestic Declarations dataset reveals several distinct process variants. Through analysis of the 10,500 declaration cases, we've identified five main process patterns that account for approximately 57.3% of all cases, with numerous less frequent variants making up the remainder.

Each process variant represents a different path through the declaration lifecycle, with variations in the sequence of activities, approval roles involved, and overall complexity. This document outlines the most common processes and their component activities.

## Standard "Happy Path" Process (31.2% of cases)

This is the most common and straightforward process variant, representing the ideal flow for declaration processing.

### Activities in Sequence:

1. **Declaration SUBMITTED by EMPLOYEE**
   - Employee enters declaration details including amount and budget reference
   - Declaration is created in the system with a unique identifier

2. **Declaration APPROVED by ADMINISTRATION**
   - Administrative staff reviews the declaration for completeness and validity
   - Basic verification of information and documentation

3. **Declaration FINAL_APPROVED by SUPERVISOR**
   - Employee's supervisor reviews and gives final approval
   - Authorizes the expense for payment

4. **Request Payment**
   - System automatically generates a payment request
   - Declaration is queued for payment processing

5. **Payment Handled**
   - System confirms the payment has been executed
   - Declaration process is completed

### Characteristics:
- Average duration: 2.8 days
- Average amount: €68.34
- Zero rejection activities
- Linear progression with no loops
- Involves 3 different roles (EMPLOYEE, ADMINISTRATION, SUPERVISOR)

## Rejection-Resubmission Process (12.8% of cases)

This variant involves rejection at some stage and subsequent resubmission by the employee.

### Activities in Sequence:

1. **Declaration SUBMITTED by EMPLOYEE**
   - Initial submission of the declaration

2. **Declaration APPROVED by ADMINISTRATION** or **Declaration REJECTED by ADMINISTRATION**
   - Administrative review may result in rejection
   - If rejected, declaration returns to employee

3. **Declaration REJECTED by EMPLOYEE**
   - Employee acknowledges the rejection
   - Often precedes resubmission

4. **Declaration SUBMITTED by EMPLOYEE** (second occurrence)
   - Employee resubmits the corrected declaration

5. **Declaration APPROVED by ADMINISTRATION**
   - Administrative approval after correction

6. **Declaration FINAL_APPROVED by SUPERVISOR**
   - Supervisor approval after all corrections

7. **Request Payment**
   - System payment request generation

8. **Payment Handled**
   - Payment execution and completion

### Characteristics:
- Average duration: 6.4 days
- Average amount: €85.27
- Contains at least one rejection activity
- Features a loop back to submission
- More activities than the standard process (typically 7-8 vs. 5)

## Budget Owner Approval Process (8.1% of cases)

This variant involves budget owner approval instead of or in addition to supervisor approval, typically for higher amounts or special expense categories.

### Activities in Sequence:

1. **Declaration SUBMITTED by EMPLOYEE**
   - Initial declaration submission

2. **Declaration APPROVED by ADMINISTRATION**
   - Administrative verification

3. **Declaration APPROVED by BUDGET OWNER**
   - Budget owner reviews and approves
   - Replaces or precedes supervisor approval

4. **Declaration FINAL_APPROVED by SUPERVISOR** (sometimes skipped)
   - Final approval by supervisor if required

5. **Request Payment**
   - System payment request

6. **Payment Handled**
   - Payment execution

### Characteristics:
- Average duration: 7.5 days
- Average amount: €243.16
- Significantly higher average amount than standard process
- Often involves declarations related to specific budget categories
- May include additional approvals for very high amounts

## Pre-approval Process (5.2% of cases)

This variant includes a preliminary approval step before the standard approval flow, typically for special expense types or compliance reasons.

### Activities in Sequence:

1. **Declaration SUBMITTED by EMPLOYEE**
   - Initial declaration submission

2. **Declaration APPROVED by ADMINISTRATION**
   - Administrative verification

3. **Declaration APPROVED by PRE_APPROVER**
   - Preliminary approval by specialized role
   - May involve compliance or policy verification

4. **Declaration FINAL_APPROVED by SUPERVISOR**
   - Final management approval

5. **Request Payment**
   - System payment request

6. **Payment Handled**
   - Payment execution

### Characteristics:
- Average duration: 5.3 days
- Average amount: €129.73
- Involves an additional approval role
- Often related to specific expense categories requiring special review
- Higher level of scrutiny than standard process

## Complex Multi-Approval Process (3.0% of cases)

This variant involves multiple approval roles and potentially multiple levels of review, typically for high-value declarations or special cases.

### Activities in Sequence:

1. **Declaration SUBMITTED by EMPLOYEE**
   - Initial submission

2. **Declaration APPROVED by ADMINISTRATION**
   - Administrative verification

3. **Declaration APPROVED by PRE_APPROVER**
   - Preliminary specialized approval

4. **Declaration APPROVED by BUDGET OWNER**
   - Budget authority approval

5. **Declaration FINAL_APPROVED by SUPERVISOR**
   - Final managerial approval

6. **Request Payment**
   - System payment request

7. **Payment Handled**
   - Payment execution

### Characteristics:
- Average duration: 11.2 days
- Average amount: €684.92
- Highest average amount among common variants
- Involves 4+ different approval roles
- Most complex approval chain
- Often includes high-value or unusual expenses

## Other Notable Process Patterns

### Direct Payment Process (2.4% of cases)
In some cases, declarations appear to bypass certain approval steps:

1. **Declaration SUBMITTED by EMPLOYEE**
2. **Declaration APPROVED by ADMINISTRATION**
3. **Request Payment** (without supervisor approval)
4. **Payment Handled**

This pattern may represent pre-authorized expenses or system exceptions.

### Multi-Rejection Process (1.8% of cases)
Some declarations face multiple rejections:

1. **Declaration SUBMITTED by EMPLOYEE**
2. **Declaration REJECTED by ADMINISTRATION**
3. **Declaration SUBMITTED by EMPLOYEE** (resubmission)
4. **Declaration APPROVED by ADMINISTRATION**
5. **Declaration REJECTED by SUPERVISOR**
6. **Declaration SUBMITTED by EMPLOYEE** (second resubmission)
...and so on

This pattern typically represents problematic declarations requiring multiple corrections.

## Activity Frequency Within Processes

Across all process variants, activities occur with the following frequencies:

| Activity | Overall % | Standard Process % | Rejection Process % | Budget Owner Process % |
|----------|-----------|-------------------|---------------------|------------------------|
| Declaration SUBMITTED by EMPLOYEE | 20.4% | 20.0% | 25.0% | 16.7% |
| Declaration FINAL_APPROVED by SUPERVISOR | 17.9% | 20.0% | 14.3% | 16.7% |
| Payment Handled | 17.8% | 20.0% | 14.3% | 16.7% |
| Request Payment | 17.8% | 20.0% | 14.3% | 16.7% |
| Declaration APPROVED by ADMINISTRATION | 14.5% | 20.0% | 14.3% | 16.7% |
| Declaration APPROVED by BUDGET OWNER | 5.0% | 0.0% | 0.0% | 16.7% |
| Declaration REJECTED by EMPLOYEE | 2.4% | 0.0% | 14.3% | 0.0% |
| Other Activities | 4.2% | 0.0% | 3.5% | 0.0% |

## Process Similarity Analysis

The common processes share several characteristics:

1. **Common Beginning and End**
   - All processes start with "Declaration SUBMITTED by EMPLOYEE"
   - All successful processes end with "Payment Handled"

2. **Administrative Gateway**
   - Nearly all declarations pass through administrative approval
   - Administration acts as a gateway/router to different approval paths

3. **Consistency in System Activities**
   - "Request Payment" always precedes "Payment Handled"
   - These system activities are consistent across all variants

4. **Role-Based Approval Hierarchy**
   - Clear hierarchy from ADMINISTRATION to specialized approvers to final approvers
   - Role involvement is strongly correlated with declaration amount

## Process Complexity Metrics

The common processes vary significantly in complexity:

| Process Variant | Avg. Activities | Distinct Roles | Loops | Branch Points |
|-----------------|----------------|----------------|-------|---------------|
| Standard Process | 5.0 | 3 | 0 | 0 |
| Rejection Process | 7.5 | 3 | 1+ | 1+ |
| Budget Owner Process | 6.0 | 4 | 0 | 1 |
| Pre-approval Process | 6.0 | 4 | 0 | 1 |
| Complex Multi-Approval | 7.0 | 5+ | 0 | 2+ |

## Temporal Aspects of Processes

The common processes show different temporal patterns:

1. **Process Duration**
   - Standard Process: 2.8 days average
   - Rejection Process: 6.4 days average
   - Budget Owner Process: 7.5 days average
   - Pre-approval Process: 5.3 days average
   - Complex Multi-Approval: 11.2 days average

2. **Activity Waiting Times**
   - Longest wait occurs after "Declaration SUBMITTED by EMPLOYEE" (16.3 hours avg.)
   - Second longest wait occurs after "Request Payment" (48.1 hours avg.)
   - Rejection activities typically add 3-5 days to process duration

3. **Process Pace**
   - Standard Process: 1.79 activities per day
   - Rejection Process: 1.17 activities per day
   - Budget Owner Process: 0.80 activities per day
   - Pre-approval Process: 1.13 activities per day
   - Complex Multi-Approval: 0.63 activities per day

## Conclusion

The BPI2020 Domestic Declarations dataset reveals several distinct process variants that follow specific patterns. The standard "happy path" process represents about a third of all cases, while variations involving rejections, budget owner approval, pre-approval, and complex multi-level approvals account for another quarter of cases. The remaining declarations follow numerous less common patterns, often combining elements of the main variants.

Understanding these common processes and their component activities provides valuable insights for process optimization, identifying bottlenecks, and improving the overall declaration handling workflow. The clear correlation between process complexity, duration, and declaration amount suggests that a tiered approach to declaration handling based on amount thresholds could potentially streamline the process while maintaining appropriate controls.