import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class InternshipSearchEngine:
    def __init__(self, data_path="database.csv"):
        self.df = pd.read_csv(data_path).fillna("")
        self.df["combined_text"] = self.df.apply(
            lambda r: (
                f"Major: {r['major']}. Role: {r['job title']} at {r['company']}. "
                f"Description: {r['job description']} {r['job duties']}. "
                f"Preparation: {r['pre-internship preparation']}. "
                f"Technical skills: {r['technical skills']}. Tools: {r['tools and software']}. "
                f"Soft skills: {r['soft skills']}. Courses: {r['courses mentioned']}. "
                f"Learned: {r['what have I learned']}. Advice: {r['advice for future interns']}."
            ),
            axis=1,
        )

        # 1. Lexical sparse vectorizers (always available offline)
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

        # 2. Dense embedding setup (OpenAI API -> local SentenceTransformer -> None)
        self.dense_backend = "none"
        self.dense_matrix = None
        self.openai_client = None
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        self.local_st_model = None

        self._init_dense_embeddings()

    def _init_dense_embeddings(self):
        texts = self.df["combined_text"].tolist()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

        # Priority 1: OpenAI Dense Embeddings
        if api_key and OpenAI is not None:
            try:
                kwargs = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                self.openai_client = OpenAI(**kwargs)
                resp = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                )
                vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                self.dense_matrix = vecs / np.maximum(norms, 1e-12)
                self.dense_backend = f"openai:{self.embedding_model}"
                return
            except Exception:
                self.openai_client = None
                self.dense_matrix = None

        # Priority 2: Local SentenceTransformer Dense Embeddings (if installed)
        if SentenceTransformer is not None:
            try:
                local_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
                self.local_st_model = SentenceTransformer(local_name)
                vecs = self.local_st_model.encode(texts, normalize_embeddings=True)
                self.dense_matrix = np.array(vecs, dtype=np.float32)
                self.dense_backend = f"local:{local_name}"
                return
            except Exception:
                self.local_st_model = None
                self.dense_matrix = None

        self.dense_backend = "none"

    def _encode_query_dense(self, query_str: str):
        if self.dense_matrix is None or not query_str:
            return None
        try:
            if self.openai_client is not None:
                resp = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=[query_str],
                )
                q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
                q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
                return q_vec
            if self.local_st_model is not None:
                q_vec = self.local_st_model.encode([query_str], normalize_embeddings=True)[0]
                return np.array(q_vec, dtype=np.float32)
        except Exception:
            return None
        return None

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

    def _compute_hybrid_similarity(self, query_str: str) -> tuple[np.ndarray, bool]:
        w_vec = self.word_vectorizer.transform([query_str])
        c_vec = self.char_vectorizer.transform([query_str])
        w_sim = cosine_similarity(w_vec, self.word_matrix).flatten()
        c_sim = cosine_similarity(c_vec, self.char_matrix).flatten()
        lexical_sim = 0.75 * w_sim + 0.25 * c_sim

        q_dense = self._encode_query_dense(query_str)
        if q_dense is not None and self.dense_matrix is not None:
            raw_dense = np.dot(self.dense_matrix, q_dense)
            # Rescale cosine similarity from typical [0.15, 0.85] text range to [0, 1]
            dense_sim = np.clip((raw_dense - 0.15) / 0.70, 0.0, 1.0)
            return 0.65 * dense_sim + 0.35 * lexical_sim, True

        return lexical_sim, False

    def search_nlp(self, query: str, major: str = "", tech_skill: str = "", top_k: int = 6, candidate_limit: int = 12):
        """Retrieve candidates with strict filters + Dense/Lexical hybrid similarity."""
        query_str = (query or "").strip()
        major_str = (major or "").strip()
        skill_terms = [t.strip().lower() for t in re.split(r"[,;/]+", tech_skill or "") if t.strip()]

        if not query_str and not major_str and not skill_terms:
            return self.structured_filter(major="", tech_skill="", top_k=candidate_limit)

        sim_scores, used_dense = (
            self._compute_hybrid_similarity(query_str)
            if query_str
            else (np.zeros(len(self.df)), False)
        )
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

            min_threshold = 0.08 if used_dense else 0.03
            if query_str and not major_str and not skill_terms and raw_sim < min_threshold:
                continue

            score = 0.0
            if query_str:
                score += 65.0 * rel_sim
            if major_str:
                score += 20.0 if not query_str else 18.0
            if skill_terms:
                ratio = len(matched_skills) / len(skill_terms)
                score += (20.0 if not query_str else 17.0) * ratio

            if score > 0 and str(row.get("advice for future interns", "")).strip():
                score = min(99.0, score + 1.5)

            if score <= 0:
                continue

            scored_items.append((score, idx, matched_skills))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx, matched_skills in scored_items[:candidate_limit]:
            row_dict = self._sanitize_record(self.df.iloc[idx].to_dict())
            row_dict["match_score"] = round(min(99.0, max(1.0, score)), 1)
            row_dict["rationale"] = self._generate_rationale(
                query=query_str,
                major=major_str,
                matched_skills=matched_skills,
                row=row_dict,
                used_dense=used_dense,
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

    def _generate_rationale(
        self,
        query: str,
        major: str,
        matched_skills: list[str],
        row: dict,
        used_dense: bool = False,
    ) -> str:
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
            reasons.append("dense + lexical semantic match" if used_dense else "semantic match to query")

        prefix = f"Relevant {title} at {company}"
        if reasons:
            prefix += f" ({'; '.join(reasons)})"
        if skills:
            prefix += f" — Technical skills: {skills[:105]}"
        if courses and courses.lower() != "nan":
            prefix += f" | Courses: {courses[:75]}"
        return prefix + "."