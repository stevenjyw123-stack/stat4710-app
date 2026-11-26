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
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

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
        if not str(cleaned.get("job title", "")).strip():
            cleaned["job title"] = "Internship Role"
        if not str(cleaned.get("major", "")).strip():
            cleaned["major"] = "Unspecified Major"
        cleaned["advice_clean"] = InternshipSearchEngine._clean_advice(cleaned)
        return cleaned

    def search_nlp(self, query: str, major: str = "", tech_skill: str = "", top_k: int = 6):
        """Natural language search with percentage match scoring."""
        query_str = (query or "").strip()
        if not query_str and not (major or tech_skill):
            return self.structured_filter(major=major, tech_skill=tech_skill, top_k=top_k)

        base_query = query_str if query_str else f"{major} {tech_skill}".strip()
        query_vec = self.vectorizer.transform([base_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        max_sim = float(similarities.max()) if similarities.max() > 0 else 1.0
        norm_sim = similarities / max_sim  # scale best semantic hit to [0, 1]

        combined_scores = np.zeros(len(self.df))
        for idx, row in self.df.iterrows():
            score = norm_sim[idx] * 55.0
            if major and major.lower() in str(row["major"]).lower():
                score += 22.0
            if tech_skill:
                skill_blob = f"{row['technical skills']} {row['tools and software']}".lower()
                if tech_skill.lower() in skill_blob:
                    score += 22.0
            if not major and not tech_skill:
                score = 50.0 + norm_sim[idx] * 46.0
            combined_scores[idx] = score

        top_indices = np.argsort(combined_scores)[::-1]
        results = []
        for idx in top_indices:
            row_dict = self._sanitize_record(self.df.iloc[idx].to_dict())
            if major and major.lower() not in row_dict["major"].lower() and not query_str:
                continue
            if tech_skill:
                skill_blob = f"{row_dict.get('technical skills', '')} {row_dict.get('tools and software', '')}".lower()
                if tech_skill.lower() not in skill_blob and not query_str:
                    continue

            pct = round(min(98.5, max(52.0, float(combined_scores[idx]))), 1)
            row_dict["match_score"] = pct
            row_dict["rationale"] = self._generate_rationale(query_str, row_dict)
            results.append(row_dict)
            if len(results) >= top_k:
                break

        return results

    def structured_filter(self, major: str = None, tech_skill: str = None, top_k: int = 6):
        """Structured filtering with percentage scoring."""
        scores = np.zeros(len(self.df))

        for idx, row in self.df.iterrows():
            score = 0.0
            major_match = bool(major and major.lower() in str(row["major"]).lower())
            skill_blob = f"{row['technical skills']} {row['tools and software']}".lower()
            skill_match = bool(tech_skill and tech_skill.lower() in skill_blob)

            if major_match:
                score += 48.0
            if skill_match:
                score += 48.0
            if not major and not tech_skill:
                score += 75.0
            scores[idx] = score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            row_dict = self._sanitize_record(self.df.iloc[idx].to_dict())
            row_dict["match_score"] = round(score, 1)
            m_flag = "Yes" if (major and major.lower() in row_dict["major"].lower()) else ("Any" if not major else "No")
            s_flag = "Yes" if (
                tech_skill
                and tech_skill.lower() in f"{row_dict.get('technical skills', '')} {row_dict.get('tools and software', '')}".lower()
            ) else ("Any" if not tech_skill else "No")
            row_dict["rationale"] = f"Criteria match — Major: {m_flag}, Technical Skill/Tool: {s_flag}."
            results.append(row_dict)
        return results

    def _generate_rationale(self, query: str, row: dict) -> str:
        skills = row.get("technical skills", "").strip()
        courses = row.get("courses mentioned", "").strip()
        company = row.get("company", "Host Organization")
        title = row.get("job title", "Internship Role")
        parts = [f"Relevant {title} experience at {company}"]
        if skills:
            parts.append(f"involving {skills[:110]}")
        if courses and courses.lower() != "nan":
            parts.append(f"(courses noted: {courses[:80]})")
        return " ".join(parts) + "."