"""Node implementations for the essay workflow."""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import EssayState
from tools import get_search_tool



PLAN_PROMPT = """You are an expert academic essay planner.
Your job is to create a clear, structured outline for an essay.

The outline should include:
- A clear thesis statement
- 3-4 main arguments / sections with brief bullet points for each
- A note on what kind of evidence / examples would strengthen each section

Be specific and actionable. The outline will be handed to a writer."""


def plan_node(state: EssayState, model) -> dict:
    print("\n📋 [Planner] Creating essay outline...")

    response = model.invoke([
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=f"Create a detailed essay outline for this topic:\n\n{state['task']}")
    ])

    print(f"   ✓ Plan created ({len(response.content)} chars)")
    return {
        "plan": response.content,
        "messages": [AIMessage(content=f"[PLAN]\n{response.content}")]
    }



RESEARCH_PROMPT = """You are a research strategist.
Given an essay topic and outline, generate exactly 3 specific, targeted
web search queries that will gather the most useful evidence and data.

Rules:
- Each query should target different aspects (e.g. statistics, examples, recent events)
- Keep each query short and precise (5-8 words max)
- Think like a journalist seeking hard facts, not vague overviews

Return ONLY the 3 queries, one per line, no numbering, no extra text."""


def research_node(state: EssayState, model) -> dict:
    print("\n🔍 [Researcher] Gathering information from the web...")

    search_tool = get_search_tool(max_results=3)

    queries_response = model.invoke([
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=(
            f"Essay topic: {state['task']}\n\n"
            f"Essay outline:\n{state['plan']}\n\n"
            "Generate 3 targeted search queries."
        ))
    ])

    raw_queries = queries_response.content.strip().split("\n")
    queries = [q.strip() for q in raw_queries if q.strip()][:3]
    print(f"   Search queries: {queries}")

    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"   🌐 Search {i}/3: '{query}'")
        try:
            results = search_tool.invoke(query)
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        all_results.append({
                            "query": query,
                            "url": r.get("url", ""),
                            "content": r.get("content", "")[:600]  
                        })
            elif isinstance(results, str):
                all_results.append({"query": query, "url": "", "content": results[:600]})
        except Exception as e:
            print(f"   ⚠️  Search failed for '{query}': {e}")
            all_results.append({"query": query, "url": "", "content": f"Search unavailable: {e}"})

    formatted_research = ""
    for idx, item in enumerate(all_results, 1):
        formatted_research += (
            f"\n--- Source {idx} (Query: '{item['query']}') ---\n"
            f"URL: {item['url']}\n"
            f"{item['content']}\n"
        )

    print(f"   ✓ Gathered {len(all_results)} sources")
    return {
        "research": formatted_research,
        "messages": [AIMessage(content=f"[RESEARCH]\nGathered {len(all_results)} sources for queries: {queries}")]
    }



DRAFT_PROMPT = """You are an expert essay writer with a talent for clear,
compelling, evidence-based writing.

Using the provided outline and research, write a complete, polished essay.

Structure requirements:
- Strong opening paragraph with a clear, arguable thesis statement
- Well-developed body paragraphs (one main idea each), each with:
  • Topic sentence
  • Supporting evidence or example (cite sources briefly e.g. "According to [URL]...")
  • Analysis connecting evidence to the thesis
- Smooth transitions between paragraphs
- A conclusion that synthesises the main points and reinforces the thesis

Style: Clear, academic but not dry. Aim for 600-900 words.
Do NOT use bullet points in the essay itself — write in flowing prose."""


def draft_node(state: EssayState, model) -> dict:
    revision = state.get("revision_num", 0)
    label = "first draft" if revision == 0 else f"draft (revision {revision})"
    print(f"\n✍️  [Writer] Writing {label}...")

    human_context = ""
    if state.get("human_feedback"):
        human_context = f"\n\nIMPORTANT — Human feedback to incorporate:\n{state['human_feedback']}"

    response = model.invoke([
        SystemMessage(content=DRAFT_PROMPT),
        HumanMessage(content=(
            f"Topic: {state['task']}\n\n"
            f"Outline:\n{state['plan']}\n\n"
            f"Research:\n{state['research']}"
            f"{human_context}"
        ))
    ])

    print(f"   ✓ Draft written ({len(response.content.split())} words approx)")
    return {
        "draft": response.content,
        "revision_num": revision + 1,
        "messages": [AIMessage(content=f"[DRAFT v{revision+1}]\n{response.content}")]
    }



CRITIQUE_PROMPT = """You are a demanding but fair academic editor.
Your job is to make this essay significantly better through honest critique.

Evaluate the essay on these dimensions:
1. THESIS CLARITY — Is the argument clear and specific? Or vague and obvious?
2. EVIDENCE QUALITY — Are claims backed with specific facts/data/examples?
3. LOGICAL FLOW — Does each paragraph connect clearly to the thesis?
4. WRITING QUALITY — Are there weak sentences, repetition, or passive-voice overuse?
5. CONCLUSION STRENGTH — Does it synthesise, not just repeat?

Format your critique as:
- One paragraph overall assessment (2-3 sentences)
- Numbered list of 4-6 SPECIFIC issues with concrete suggestions for each

Be direct. Name specific sentences or paragraphs that are weak.
Do NOT say "good job" or soften criticism — the writer needs to improve."""


def critique_node(state: EssayState, model) -> dict:
    print(f"\n🧐 [Critic] Reviewing draft {state.get('revision_num', 1)}...")

    response = model.invoke([
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=(
            f"Essay topic: {state['task']}\n\n"
            f"Essay draft to critique:\n\n{state['draft']}"
        ))
    ])

    print(f"   ✓ Critique ready ({len(response.content)} chars)")
    return {
        "critique": response.content,
        "messages": [AIMessage(content=f"[CRITIQUE]\n{response.content}")]
    }



REVISION_PROMPT = """You are a skilled essay reviser. You will receive:
- The original essay draft
- Critique pointing out specific problems
- (Possibly) additional human feedback

Your job: Substantially rewrite the essay to address EVERY critique point.
Do not make cosmetic changes — make real structural and content improvements.

Requirements:
- Keep what works, fix what doesn't
- Add specific evidence where the critique says evidence is weak
- Fix logical gaps
- Improve weak sentences / transitions
- Maintain the same approximate word count (600-900 words)
- Write in flowing prose, no bullet points"""


def revision_node(state: EssayState, model) -> dict:
    rev_num = state.get("revision_num", 1)
    print(f"\n✏️  [Reviser] Applying revisions (cycle {rev_num})...")

    human_section = ""
    if state.get("human_feedback"):
        human_section = f"\n\nADDITIONAL HUMAN FEEDBACK (prioritise this):\n{state['human_feedback']}"

    response = model.invoke([
        SystemMessage(content=REVISION_PROMPT),
        HumanMessage(content=(
            f"Topic: {state['task']}\n\n"
            f"Research (use this to add missing evidence):\n{state['research']}\n\n"
            f"Current draft:\n{state['draft']}\n\n"
            f"Critique to address:\n{state['critique']}"
            f"{human_section}"
        ))
    ])

    print(f"   ✓ Revision complete ({len(response.content.split())} words approx)")
    return {
        "draft": response.content,
        "human_feedback": "",
        "revision_num": rev_num + 1,
        "messages": [AIMessage(content=f"[REVISION {rev_num}]\n{response.content}")]
    }



FINAL_PROMPT = """You are a professional copy editor doing a final polish.
The essay is structurally sound — your job is a light cleanup:
- Fix any grammar or punctuation errors
- Smooth any awkward phrasing
- Ensure the opening line is punchy and engaging
- Ensure the final sentence is memorable and strong
- Do NOT change the content, arguments, or structure

Return the full polished essay only — no commentary."""


def final_node(state: EssayState, model) -> dict:
    """
    Optional final polish pass — light edits only, no structural changes.
    """
    print("\n✨ [Finaliser] Final polish pass...")

    response = model.invoke([
        SystemMessage(content=FINAL_PROMPT),
        HumanMessage(content=f"Final polish this essay:\n\n{state['draft']}")
    ])

    print("   ✓ Essay finalised!")
    return {
        "draft": response.content,
        "messages": [AIMessage(content=f"[FINAL]\n{response.content}")]
    }