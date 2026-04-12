---
title: Food Cop AI
emoji: 🍱
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Food Cop AI 

reality check of food  Indian food safety powered by Llama AI.
with fssai rules and EFSA european food security authority

## what it does 
Checks food labels for banned FSSAI and EFSA additives
- Identifies health risks
- Gives brutal honest verdict: SAFE or DANGEROUS

1. Accepts product input
User can type a food product name and ingredients manually.


2. Reads the ingredient list
It parses ingredients one by one.


3. Checks risky ingredients
It compares ingredients against your banned/risky ingredient database (like additives, dyes, preservatives, etc.).


4. Detects harmful patterns
It flags ingredients associated with health concerns or restrictions.


5. Gives a safety verdict
Returns something like:

Safe ✅

Caution ⚠️

Unsafe ❌



6. Explains why
It tells which ingredients caused concern and why.


7. Shows health impact
It can mention possible risks (for example: allergy concern, high additive load, banned substance, etc.).


8. Works for multiple difficulty tasks
Useful for your hackathon tasks: easy / medium / hard validations.


9. Provides structured output
Returns machine-readable results for scoring in the OpenEnv environment.


10. Helps consumers decide quickly
Saves time by turning confusing labels into understandable safety guidance.



One-Line Pitch

An AI food label inspector that reads ingredients and instantly tells whether a product is safe, risky, or unsafe, with reasons.


