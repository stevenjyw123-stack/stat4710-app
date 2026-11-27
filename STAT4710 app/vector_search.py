import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InternshipSearchEngine:
    def __init__(self, data_path="database.csv"):
        self.df = pd.read_csv(data_path).fillna("")
        self.df["combined_text"] = self.df.apply(
            lambda r: (
                f"{r['major']} {r['job title']} {r['company']} {r['job description']} "
                f"{r['job duties']} {r['pre-internship preparation']} {r['technical skills']} "
                f"{r['soft skills']} {r['courses mentioned']} {r['what have I learned']} "
                f"{r['advice for future interns']} {r['tools and software']}"
            ),
            axis=1,
        )
        # Word + char_wb n-grams help capture technical terms & course codes while keeping offline speed
        self.word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=6000,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=8000,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.df["combined_text"])
        self.char_matrix = self.char_vectorizer.fit_transform(self.df["combined_text"])

    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        t = (term or "").strip().lower()
        hay = str(text or "").lower()
        if not t or not hay:
            return False
        if len(t) <= 2 and t.isalpha():
            return bool(re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", hay))
        return t in hay

    @staticmethod
    def _clean_advice(row: dict) -> str:
        raw = str(row.get("advice for future interns", "")).strip()
        if raw and raw.lower() not in {"nan", "none", "na", "n/a", "nil"}:
            return raw
        learned = str(row.get("what have I learned", "")).strip()
        if learned:
            return f"Key reflection from intern: {learned}"
        return "Prepare relevant foundational courses early, clarify project expectations with your team, and document your workflow."

    @staticmethod
    def _sanitize_record(row: dict) -> dict:
        cleaned = dict(row)
        cleaned.pop("ID", None)
        cleaned.pop("combined_text", None)
        if not str(cleaned.get("company", "")).strip():
            cleaned["company"] = "Host Organization (Unspecified)"
        elif str(cleaned.get("company", "")).strip().lower().startswith("d actuarial science"):
            cleaned["company"] = "Department of Statistics and Actuarial Science, HKU"
        if not str(cleaned.get("job title", "")).strip():
            cleaned["job title"] = "Internship Role"
        if not str(cleaned.get("major", "")).strip():
            cleaned["major"] = "Unspecified Major"
        cleaned["advice_clean"] = InternshipSearchEngine._clean_advice(cleaned)
        return cleaned

    def _compute_hybrid_similarity(self, query_str: str) -> np.ndarray:
        w_vec = self.word_vectorizer.transform([query_str])
        c_vec = self.char_vectorizer.transform([query_str])
        w_sim = cosine_similarity(w_vec, self.word_matrix).flatten()
        c_sim = cosine_similarity(c_vec, self.char_matrix).flatten()
        return 0.75 * w_sim + 0.25 * c_sim

    def search_nlp(self, query: str, major: str = "", tech_skill: str = "", top_k: int = 6):
        """Strict filtering + calibrated percentage match scoring without artificial floors."""
        query_str = (query or "").strip()
        major_str = (major or "").strip()
        skill_terms = [t.strip().lower() for t in re.split(r"[,;/]+", tech_skill or "") if t.strip()]

        if not query_str and not major_str and not skill_terms:
            return self.structured_filter(major="", tech_skill="", top_k=top_k)

        sim_scores = self._compute_hybrid_similarity(query_str) if query_str else np.zeros(len(self.df))
        max_sim = float(sim_scores.max()) if len(sim_scores) > 0 and sim_scores.max() > 0 else 0.0

        scored_items = []
        for idx, row in self.df.iterrows():
            major_ok = True if not major_str else self._term_in_text(major_str, row.get("major", ""))
            if not major_ok:
                continue

            skill_blob = f"{row.get('technical skills', '')} {row.get('tools and software', '')}"
            matched_skills = [t for t in skill_terms if self._term_in_text(t, skill_blob)]
            if skill_terms and not matched_skills:
                continue

            raw_sim = float(sim_scores[idx]) if query_str else 0.0
            rel_sim = (raw_sim / max_sim) if (query_str and max_sim > 0) else 0.0

            # Reject completely irrelevant queries that have near-zero similarity and no filter match
            if query_str and not major_str and not skill_terms and raw_sim < 0.03:
                continue

            score = 0.0
            if query_str:
                score += 65.0 * rel_sim
            if major_str:
                score += 20.0 if not query_str else 18.0
            if skill_terms:
                ratio = len(matched_skills) / len(skill_terms)
                score += (20.0 if not query_str else 17.0) * ratio

            # Small bonus for rich reflection text when tie-breaking
            if score > 0 and str(row.get("advice for future interns", "")).strip():
                score = min(99.0, score + 1.5)

            if score <= 0:
                continue

            scored_items.append((score, idx, matched_skills))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx, matched_skills in scored_items[:top_k]:
            row_dict = self._sanitize_record(self.df.iloc[idx].to_dict())
            row_dict["match_score"] = round(min(99.0, max(1.0, score)), 1)
            row_dict["rationale"] = self._generate_rationale(
                query=query_str,
                major=major_str,
                matched_skills=matched_skills,
                row=row_dict,
            )
            results.append(row_dict)

        return results

    def structured_filter(self, major: str = "", tech_skill: str = "", top_k: int = 6):
        """Strict structured filtering with calibrated percentage scoring."""
        major_str = (major or "").strip()
        skill_terms = [t.strip().lower() for t in re.split(r"[,;/]+", tech_skill or "") if t.strip()]

        scored_items = []
        for idx, row in self.df.iterrows():
            major_match = bool(major_str and self._term_in_text(major_str, row.get("major", "")))
            if major_str and not major_match:
                continue

            skill_blob = f"{row.get('technical skills', '')} {row.get('tools and software', '')}"
            matched_skills = [t for t in skill_terms if self._term_in_text(t, skill_blob)]
            if skill_terms and not matched_skills:
                continue

            if major_str and skill_terms:
                score = 72.0 + 24.0 * (len(matched_skills) / len(skill_terms))
            elif major_str:
                score = 84.0
            elif skill_terms:
                score = 68.0 + 22.0 * (len(matched_skills) / len(skill_terms))
            else:
                # Unfiltered browse mode: reflects general dataset match
                score = 60.0

            scored_items.append((score, idx, matched_skills))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx, matched_skills in scored_items[:top_k]:
            row_dict = self._sanitize_record(self.df.iloc[idx].to_dict())
            row_dict["match_score"] = round(min(99.0, score), 1)
            m_flag = "Matched" if major_str else "Any"
            s_flag = ", ".join(matched_skills) if matched_skills else ("Any" if not skill_terms else "None")
            row_dict["rationale"] = f"Filtered match — Major: {m_flag}; Skill/Tool: {s_flag}."
            results.append(row_dict)

        return results

    def _generate_rationale(self, query: str, major: str, matched_skills: list[str], row: dict) -> str:
        skills = row.get("technical skills", "").strip()
        courses = row.get("courses mentioned", "").strip()
        company = row.get("company", "Host Organization")
        title = row.get("job title", "Internship Role")

        reasons = []
        if major:
            reasons.append(f"major includes {major}")
        if matched_skills:
            reasons.append(f"skills match {', '.join(matched_skills)}")
        if query:
            reasons.append("semantic match to query")

        prefix = f"Relevant {title} at {company}"
        if reasons:
            prefix += f" ({'; '.join(reasons)})"
        if skills:
            prefix += f" — Technical skills: {skills[:105]}"
        if courses and courses.lower() != "nan":
            prefix += f" | Courses: {courses[:75]}"
        return prefix + "."