import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InternshipSearchEngine:
    def __init__(self, data_path='database.csv'):
        self.df = pd.read_csv(data_path).fillna('')
        self.df['combined_text'] = self.df.apply(
            lambda r: f"{r['major']} {r['job title']} {r['company']} {r['job description']} "
                      f"{r['job duties']} {r['technical skills']} {r['soft skills']} "
                      f"{r['what have I learned']} {r['tools and software']}", axis=1
        )
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])

    def search_nlp(self, query: str, top_k: int = 5):
        """Natural language search matching student queries against internships."""
        if not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = round(float(similarities[idx]) * 100, 1)
            if score <= 0:
                continue
            row = self.df.iloc[idx].to_dict()
            row['match_score'] = score
            row['rationale'] = self._generate_rationale(query, row)
            results.append(row)
        return results

    def structured_filter(self, major: str = None, tech_skill: str = None, top_k: int = 5):
        """Structured filtering with explicit point scoring."""
        scores = np.zeros(len(self.df))

        for idx, row in self.df.iterrows():
            score = 0.0
            if major and major.lower() in str(row['major']).lower():
                score += 50.0
            if tech_skill and tech_skill.lower() in str(row['technical skills']).lower():
                score += 50.0
            scores[idx] = score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score == 0:
                continue
            row = self.df.iloc[idx].to_dict()
            row['match_score'] = score
            row[
                'rationale'] = f"Matches criteria: Major match ({'Yes' if major and major.lower() in str(row['major']).lower() else 'No'}), Skill match ({'Yes' if tech_skill and tech_skill.lower() in str(row['technical skills']).lower() else 'No'})."
            results.append(row)
        return results

    def _generate_rationale(self, query: str, row: dict) -> str:
        """Concise mock/heuristic rationale generator (can be connected to OpenAI/Gemini API)."""
        skills = row.get('technical skills', 'N/A')
        company = row.get('company', 'Unknown')
        title = row.get('job title', 'Internship')
        return f"This position at {company} as {title} highly matches your query context. Key skills involved: {skills[:120]}..."