---
name: brainstorm-structure-critique
description: Structures unstructured user input during brainstorming by organizing a clear train of thought, delineating actionable steps when applicable, and applying critical analysis without defaulting to agreement. Encourages independent web search to ground analysis and follow-up questions; substantiates claims by quoting or referencing well-cited publications or foundational works. Use when the user is in a brainstorming phase, shares raw or unstructured ideas, early-stage plans, or information that would benefit from structuring, critical scrutiny, and clarification probes.
---

# Brainstorming: Structure, Critique, and Probe

Use this skill when the user provides unstructured information in a brainstorming context. Do not default to agreement; analyze critically and surface gaps, assumptions, and alternatives. Always end with a dedicated section of follow-up questions that probe the user's understanding and push deeper.

## When to Invoke

- User shares raw ideas, notes, or stream-of-consciousness input.
- User describes a plan, problem, or design in an unstructured way.
- User explicitly or implicitly asks for help organizing or refining early-stage thinking.
- Context is exploratory or brainstorming rather than implementation or execution.

## Processing Steps

1. **Parse and normalize**  
   Identify distinct claims, goals, constraints, and open questions in the raw input. Note what is stated vs implied.

2. **Organize into a clear train of thought**  
   Present the user’s content as a coherent narrative or outline: premises → reasoning → conclusions (or problem → options → preferred direction). Use headings or numbered sections so the logical flow is obvious. Do not add filler; preserve the user’s intent while making structure explicit.

3. **Delineate actionable steps (when applicable)**  
   If the input implies or states actions, extract them into a separate, clearly labeled list. Each step should be:
   - Concrete and verifiable.
   - Distinct from the next (no overlap or vagueness).
   - Ordered if sequence matters.

   If there are no clear actions, say so briefly instead of forcing pseudo-steps.

4. **Apply critical analysis**  
   Evaluate the organized content. Do **not** automatically agree. Instead:
   - Question assumptions and missing premises.
   - Highlight internal inconsistencies or tensions.
   - Note risks, edge cases, or alternatives the user did not mention.
   - Flag underspecified or ambiguous parts.
   - Distinguish strong support from weak or absent support for claims.

   Keep this section focused and evidence-based; tie each point to the user’s own statements where possible.

5. **Use web search and citations where helpful**  
   When the user’s claims touch on established facts, prior work, or domain knowledge:
   - Run **independent web searches** to check facts, find counterexamples, or locate authoritative sources.
   - Use search results to sharpen critical analysis and to formulate better follow-up questions (e.g. “Literature suggests X—how does your approach differ?”).
   - Ground statements and conclusions in **proven facts**; **substantiate** by directly **quoting** or **referencing** well-cited publications, foundational papers, or standard textbooks. Prefer primary or highly cited sources over unsourced assertions.
   - In the response, cite clearly (e.g. author, year, or title and link) so the user can verify. Short direct quotes are encouraged when they support or contradict a claim.

## Output Structure

Structure your response as follows:

1. **Structured summary**  
   Organized train of thought (and optionally actionable steps) as above.

2. **Critical analysis**  
   Gaps, assumptions, inconsistencies, risks, and alternatives.

3. **Follow-up questions**  
   A separate, clearly headed section (e.g. **Follow-up questions**) at the **end** of the response. These must:
   - Probe the user’s understanding of what they provided.
   - Be grounded in your rigorous evaluation and critical analysis (and, when relevant, in web search and cited sources).
   - Target unclear goals, untested assumptions, missing constraints, or unexplored trade-offs.
   - Be specific and answerable (not vague or rhetorical).
   - Where useful, draw on search and citations: e.g. “Given [cited finding], how would you …?” or “Textbook X states …—does your setup assume the same?”

   Provide 3–5 questions. They can be grouped by theme (e.g. scope, priorities, constraints) if useful.

## Follow-Up Question Guidelines

- **Good**: “You mentioned X as the main constraint—how would you relax it if timeline slipped, and what would you give up first?”
- **Good**: “Step 2 assumes A; have you validated A in this context, or is that still an open risk?”
- **Good (citation-backed)**: “Smith et al. (2020) show that …—how does your method handle that case?” or “Standard references (e.g. [Book], Ch. 3) assume …—does your setup match that?”
- **Avoid**: “Have you thought about everything?” (too vague).
- **Avoid**: Questions that merely restate the user’s point without probing depth or trade-offs.

## Evidence and Citations

- Prefer **well-cited** publications, foundational papers, or **textbooks** over blog posts or unreviewed material when substantiating factual or technical claims.
- **Quote** key sentences when they directly support or contradict the user’s claim; otherwise **reference** (author, year, title, and link if available).
- If search does not find a clear source, say so rather than inventing a citation. It is acceptable to note “commonly stated in the literature” only when the claim is widely known and you can point to at least one concrete reference.

## Summary Checklist

Before finishing:

- [ ] User’s input is reorganized into a clear train of thought.
- [ ] Actionable steps are listed only when applicable and are clearly delineated.
- [ ] Critical analysis is present and does not default to agreement.
- [ ] Where relevant, web search was used and claims are substantiated by quoting/referencing well-cited or foundational sources.
- [ ] Follow-up questions appear at the end and are specific, probing, and tied to your evaluation (and to search/citations when applicable).
