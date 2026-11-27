import json
import os
from openai import OpenAI


class InternshipLLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.enabled = bool(self.api_key)
        self.client = None

        if self.enabled:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)

    def verify_and_rank_candidates(
        self,
        query: str,
        major: str,
        tech_skill: str,
        candidates: list[dict],
        top_k: int = 6,
    ) -> tuple[list[dict], bool]:
        """
        Ask LLM to inspect retrieved candidates, filter irrelevant ones when query is specific,
        and refine match_score & rationale. Returns (verified_results, used_llm).
        """
        if not self.enabled or not self.client or not candidates:
            return candidates[:top_k], False

        compact_candidates = []
        for idx, item in enumerate(candidates[:12]):
            compact_candidates.append({
                "idx": idx,
                "major": item.get("major", ""),
                "job_title": item.get("job title", ""),
                "company": item.get("company", ""),
                "semester": f"{item.get('year', '')} {item.get('semester', '')}".strip(),
                "technical_skills": str(item.get("technical skills", ""))[:180],
                "tools": str(item.get("tools and software", ""))[:140],
                "courses": str(item.get("courses mentioned", ""))[:120],
                "advice": str(item.get("advice_clean", ""))[:220],
                "base_score": item.get("match_score", 0),
            })

        system_prompt = (
            "You are an academic internship advisor assistant for HKU STAT 4710. "
            "Given a student's query/filters and candidate internship records (anonymized, no names/IDs), "
            "check whether each candidate is genuinely relevant to the student's needs. "
            "Drop clearly irrelevant items unless there are very few matches. "
            "Return ONLY valid JSON with key 'items': a list of objects "
            "{'idx': int, 'relevant': bool, 'score': float (0-99), 'rationale': str}."
        )
        user_payload = {
            "query": query,
            "major_filter": major,
            "skill_filter": tech_skill,
            "top_k": top_k,
            "candidates": compact_candidates,
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            raw_items = parsed.get("items", [])

            by_idx = {c["idx"]: c for c in candidates[:12]}
            verified = []
            seen_idx = set()

            for entry in raw_items:
                c_idx = entry.get("idx")
                if c_idx not in by_idx or c_idx in seen_idx:
                    continue
                if entry.get("relevant") is False and len(raw_items) > 2:
                    continue
                seen_idx.add(c_idx)
                row_copy = dict(by_idx[c_idx])
                if isinstance(entry.get("score"), (int, float)):
                    row_copy["match_score"] = round(min(99.0, max(1.0, float(entry["score"]))), 1)
                if entry.get("rationale"):
                    row_copy["rationale"] = str(entry["rationale"]).strip()
                verified.append(row_copy)
                if len(verified) >= top_k:
                    break

            if verified:
                verified.sort(key=lambda x: x.get("match_score", 0), reverse=True)
                return verified[:top_k], True
            return candidates[:top_k], True
        except Exception:
            return candidates[:top_k], False

    def generate_guidance_reply(
        self,
        question_label: str,
        major: str,
        tech_skill: str,
        results: list[dict],
        fallback_reply: str,
    ) -> tuple[str, bool]:
        """Generate a concise, practical student-oriented guidance paragraph using LLM."""
        if not self.enabled or not self.client:
            return fallback_reply, False

        if not results:
            return fallback_reply, True

        summary_rows = [
            {
                "role": r.get("job title", ""),
                "company": r.get("company", ""),
                "major": r.get("major", ""),
                "match_score": r.get("match_score", ""),
                "courses": r.get("courses mentioned", ""),
                "advice": r.get("advice_clean", ""),
            }
            for r in results[:6]
        ]

        system_prompt = (
            "You are a helpful internship advisor for university statistics, risk/actuarial, and analytics students. "
            "Write a clear, encouraging, concise English guidance paragraph (3-5 sentences) based ONLY on the provided "
            "2025 student internship records. Mention relevant pre-internship courses if present, highlight practical "
            "preparation tips from past interns, and never invent student names, IDs, or external facts."
        )
        user_prompt = json.dumps(
            {
                "student_question": question_label,
                "major_filter": major,
                "skill_filter": tech_skill,
                "matched_records": summary_rows,
            },
            ensure_ascii=False,
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            reply = (resp.choices[0].message.content or "").strip()
            return (reply if reply else fallback_reply), True
        except Exception:
            return fallback_reply, False