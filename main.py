from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json, re
from dotenv import load_dotenv

# ============================================================
# --- ENVIRONMENT SETUP ---
# ============================================================

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
else:
    print("⚠️ LANGCHAIN_API_KEY not found — LangChain tracing disabled.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ============================================================
# --- FASTAPI APP ---
# ============================================================

app = FastAPI(title="ABAP Suggestion API - Replace Queries on T881/T881T/T882G")

# ============================================================
# --- MODELS ---
# ============================================================

class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None


class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    findings: Optional[List[Finding]] = Field(default_factory=list)

# ============================================================
# --- LLM PROMPT FOR T881/T881T/T882G QUERIES ---
# ============================================================

SYSTEM_MSG = """
You are an ABAP modernization advisor for S/4HANA syntax upgrades.
Output strictly valid JSON as response.

Rules:
- Identify SELECT queries on tables T881, T881T, or T882G.
- Suggest replacing them with corresponding CL_FINS_ACDOC_UTIL method calls:
  - T881  → CL_FINS_ACDOC_UTIL=>GET_T881_EMU
  - T881T → CL_FINS_ACDOC_UTIL=>GET_T881T_EMU
  - T882G → CL_FINS_ACDOC_UTIL=>GET_T882G_EMU
- Use only the 'suggestion' field for the actionable bullet points.
- Skip any finding without a suggestion.
- Do not include the snippet in the output.

Return JSON only with:
{
  "assessment": "<summary of T881/T881T/T882G query findings>",
  "llm_prompt": "<bulleted list of suggestions for replacement>"
}
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

Findings (JSON, filtered for T881/T881T/T882G queries):
{findings_json}

Instructions:
1. Write an 'assessment' summarizing the issue and modernization impact.
2. For each finding with a valid suggestion, create a bullet in 'llm_prompt':
   - Use only the suggestion text (skip snippet).
   - Ensure each bullet is unique and actionable.
   - Skip findings without a suggestion.

Return strictly valid JSON:
{
  "assessment": "<summary paragraph>",
  "llm_prompt": "<bullet list of actionable suggestions>"
}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ============================================================
# --- CORE LOGIC ---
# ============================================================

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    relevant = []
    for f in (unit.findings or []):
        if f.suggestion and f.suggestion.strip():
            relevant.append(f)
        elif f.snippet:
            # Auto-suggest if snippet matches T881/T881T/T882G
            if re.search(r"\bFROM\s+(T881T?|T882G)\b", f.snippet, re.IGNORECASE):
                tbl = re.search(r"\bFROM\s+(T881T?|T882G)\b", f.snippet, re.IGNORECASE).group(1).upper()
                mapping = {
                    "T881": "CL_FINS_ACDOC_UTIL=>GET_T881_EMU",
                    "T881T": "CL_FINS_ACDOC_UTIL=>GET_T881T_EMU",
                    "T882G": "CL_FINS_ACDOC_UTIL=>GET_T882G_EMU",
                }
                f.suggestion = f"Replace query on table {tbl} with a call to {mapping[tbl]}."
                relevant.append(f)

    if not relevant:
        return None

    findings_json = json.dumps([f.model_dump() for f in relevant], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "class_implementation": unit.class_implementation or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ============================================================
# --- API ENDPOINTS ---
# ============================================================

@app.post("/assess")
async def assess_t881_queries(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        llm_out = llm_assess_and_prompt(u)
        if not llm_out:
            continue
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name,
            "class_implementation": u.class_implementation,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "assessment": llm_out.get("assessment", ""),
            "llm_prompt": prompt_out
        }
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "note": "Replace_T881_T881T_T882G", "model": OPENAI_MODEL}
