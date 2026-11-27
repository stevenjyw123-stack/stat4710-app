import os
import re
from flask import Flask, jsonify, render_template, request
from dashboard_service import DashboardService
from llm_service import InternshipLLMService
from vector_search import InternshipSearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "database.csv")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

dashboard_service = DashboardService(DATA_PATH)
search_engine = InternshipSearchEngine(DATA_PATH)
llm_service = InternshipLLMService()


def _extract_course_codes(text: str) -> list[str]:
    if not text:
        return []
    return sorted({c.replace(" ", "") for c in re.findall(r"\b(?:[A-Z]{2,5}\s*)?\d{4}\b", str(text).upper())})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard/stats", methods=["GET"])
def get_dashboard_stats():
    """REST API endpoint to feed analytics dashboard."""
    try:
        summary = dashboard_service.get_summary_stats()
        charts = dashboard_service.get_chart_data()
        return jsonify({
            "summary": summary,
            "charts": charts,
            "engine_info": {
                "dense_backend": search_engine.dense_backend,
                "llm_enabled": llm_service.enabled,
                "llm_model": llm_service.model if llm_service.enabled else "fallback-local",
            },
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load dashboard stats: {str(e)}"}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend_internships():
    """REST API endpoint with Dense/Hybrid retrieval + LLM verification + LLM guidance."""
    try:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        major = (data.get("major") or "").strip()
        tech_skill = (data.get("tech_skill") or "").strip()
        mode = data.get("mode", "nlp")
        top_k = max(1, min(6, int(data.get("top_k") or 6)))

        if mode == "filter" and not query:
            candidates = search_engine.structured_filter(major=major, tech_skill=tech_skill, top_k=12)
        else:
            candidates = search_engine.search_nlp(
                query=query,
                major=major,
                tech_skill=tech_skill,
                top_k=top_k,
                candidate_limit=12,
            )

        results, used_llm_rank = llm_service.verify_and_rank_candidates(
            query=query,
            major=major,
            tech_skill=tech_skill,
            candidates=candidates,
            top_k=top_k,
        )

        all_courses = []
        for r in results:
            all_courses.extend(_extract_course_codes(r.get("courses mentioned", "")))
        top_matched_courses = list(dict.fromkeys(all_courses))[:8]

        lead_advice = results[0]["advice_clean"] if results else (
            "No matching records found for this combination. Try broadening the query, choosing 'All Majors', or clearing the skill filter."
        )
        echoed_question = query or f"Major: {major or 'All'} | Skill: {tech_skill or 'All'}"
        fallback_reply = (
            f"Found {len(results)} matching internship records from the 2025 dataset"
            + (f" (courses often noted in these matches: {', '.join(top_matched_courses)})" if top_matched_courses else "")
            + f". Key advice for interns: {lead_advice}"
        )

        final_reply, used_llm_reply = llm_service.generate_guidance_reply(
            question_label=echoed_question,
            major=major,
            tech_skill=tech_skill,
            results=results,
            fallback_reply=fallback_reply,
        )

        return jsonify({
            "question": echoed_question,
            "assistant_reply": final_reply,
            "llm_active": bool(used_llm_rank or used_llm_reply),
            "dense_backend": search_engine.dense_backend,
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        return jsonify({
            "question": "",
            "assistant_reply": f"Error processing recommendation request: {str(e)}",
            "llm_active": False,
            "dense_backend": search_engine.dense_backend,
            "count": 0,
            "results": [],
        }), 500


if __name__ == "__main__":
    print(f"STAT 4710 Web App running at http://127.0.0.1:5005 | Dense={search_engine.dense_backend} | LLM={llm_service.enabled}")
    app.run(debug=True, port=5005)