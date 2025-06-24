# Hallucination Log

This document tracks instances where the math tutor provided incorrect, misleading, or fabricated information.

## 📋 Log Format

For each hallucination incident, record:
- **Date/Time**: When the incident occurred
- **Query**: The student's original question
- **Strategy Used**: Which prompt strategy was active
- **Hallucinated Response**: The incorrect information provided
- **Correct Information**: What the response should have been
- **Severity**: Low/Medium/High impact on student learning
- **Type**: Mathematical error, conceptual confusion, fabricated facts, etc.

---

## 🚨 Hallucination Incidents

### [Date: YYYY-MM-DD HH:MM]
**Query**: [Student's question]
**Strategy**: [zero_shot/few_shot/cot_prompt/meta_prompt]
**Hallucinated Response**: 
```
[Copy the incorrect response here]
```
**Correct Information**: 
```
[Provide the accurate information]
```
**Severity**: [Low/Medium/High]
**Type**: [Mathematical error/Conceptual confusion/Fabricated facts/Other]
**Notes**: [Additional context or analysis]

---

## 📊 Hallucination Analysis

### Common Patterns
[To be filled as patterns emerge]

### Most Problematic Areas
[To be filled based on logged incidents]

### Strategy-Specific Issues
- **Zero-shot**: [Common hallucination types]
- **Few-shot**: [Common hallucination types]
- **Chain-of-Thought**: [Common hallucination types]
- **Meta-prompt**: [Common hallucination types]

### Severity Distribution
- **High Severity**: [Count] incidents
- **Medium Severity**: [Count] incidents  
- **Low Severity**: [Count] incidents

## 🔧 Mitigation Strategies

### Identified Solutions
[To be filled based on analysis]

### Prompt Improvements
[Specific changes to reduce hallucinations]

### Validation Checks
[Additional verification steps to implement]

---

## 📝 Notes for Evaluators

- Mark any response with mathematical errors as a hallucination
- Pay special attention to:
  - Incorrect formulas or procedures
  - Made-up mathematical rules
  - Wrong numerical calculations
  - Conceptual misunderstandings
  - Fabricated examples or references

- Rate hallucination severity based on potential impact:
  - **High**: Could seriously mislead student understanding
  - **Medium**: Noticeable error but limited impact
  - **Low**: Minor error with minimal learning impact
