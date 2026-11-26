import re
from collections import Counter
import pandas as pd


class DashboardService:
    def __init__(self, data_path="database.csv"):
        self.df = pd.read_csv(data_path).fillna("")
        self.df["_courses_list"] = self.df["courses mentioned"].apply(self._extract_courses)
        self.df["_skills_list"] = self.df.apply(self._extract_skills, axis=1)
        self.df["_soft_skills_list"] = self.df["soft skills"].apply(self._extract_soft_skills)

    @staticmethod
    def _extract_courses(text: str) -> list[str]:
        if not str(text).strip():
            return []
        found = re.findall(r"\b(?:[A-Z]{2,5}\s*)?\d{4}\b", str(text).upper())
        return sorted({c.replace(" ", "") for c in found})

    @staticmethod
    def _extract_skills(row) -> list[str]:
        blob = f"{row.get('technical skills', '')} {row.get('tools and software', '')}".lower()
        skill_patterns = [
            ("Excel", r"\bexcel\b"),
            ("R", r"\br\b"),
            ("VBA", r"\bvba\b"),
            ("Python", r"\bpython\b"),
            ("SQL", r"\bsql\b"),
            ("Prophet", r"\bprophet\b"),
            ("IFRS 17", r"\bifrs\s*17\b"),
            ("Power BI", r"\bpower\s*bi\b"),
            ("SAS", r"\bsas\b"),
            ("RAG", r"\brag\b"),
            ("AI Agent", r"\bagent"),
            ("Tableau", r"\btableau\b"),
        ]
        return [label for label, pat in skill_patterns if re.search(pat, blob)]

    @staticmethod
    def _extract_soft_skills(text: str) -> list[str]:
        blob = str(text).lower()
        if not blob.strip():
            return []
        soft_patterns = [
            ("Communication", r"communicat"),
            ("Teamwork & Collaboration", r"teamwork|collaborat|teammate"),
            ("Presentation", r"presentat"),
            ("Time Management", r"time management|deadline|prioritiz"),
            ("Problem Solving", r"problem[- ]solv|troubleshoot"),
            ("Attention to Detail", r"detail|accuracy|careful"),
            ("Stakeholder / Client Mgmt", r"client|stakeholder|cross-functional"),
            ("Adaptability & Learning", r"adapt|fast learn|self-learn|independ"),
            ("Documentation & Reporting", r"document|report writing"),
            ("Project Management", r"project management|coordinat"),
        ]
        return [label for label, pat in soft_patterns if re.search(pat, blob)]

    def get_summary_stats(self) -> dict:
        """Metric cards summary data (2025 dataset)."""
        companies_clean = [str(c).strip() for c in self.df["company"] if str(c).strip()]
        major_series = self.df["major"].apply(
            lambda x: str(x).strip() if str(x).strip() else "Unspecified Major"
        )
        majors_breakdown = [
            {"name": name, "count": int(cnt)}
            for name, cnt in major_series.value_counts().items()
        ]

        return {
            "year": "2025",
            "total_records": int(len(self.df)),
            "total_companies": int(len(set(companies_clean))),
            "named_records": int(len(companies_clean)),
            "total_majors": int(major_series.nunique()),
            "total_courses": int(self.df["course"].replace("", pd.NA).dropna().nunique()),
            "majors_breakdown": majors_breakdown,
        }

    def get_chart_data(self) -> dict:
        """Charts data for dashboard summary tab."""
        # 1. Top 10 Companies
        companies_series = self.df["company"].apply(
            lambda x: str(x).strip() if str(x).strip() else ""
        )
        top_companies = companies_series[companies_series != ""].value_counts().head(10)
        companies_data = [{"name": str(k), "count": int(v)} for k, v in top_companies.items()]

        # 2. Top 10 Technical Skills & Tools
        skill_counter = Counter()
        for skills in self.df["_skills_list"]:
            for s in skills:
                skill_counter[s] += 1
        top_tech_skills = [{"name": k, "count": int(v)} for k, v in skill_counter.most_common(10)]

        # 3. Top 10 Useful Pre-Internship Courses
        course_counter = Counter()
        for courses in self.df["_courses_list"]:
            for c in courses:
                course_counter[c] += 1
        top_courses = [{"name": k, "count": int(v)} for k, v in course_counter.most_common(10)]

        # 4. Top 10 Soft Skills Demand (replaces sparse duration chart)
        soft_counter = Counter()
        for s_list in self.df["_soft_skills_list"]:
            for s in s_list:
                soft_counter[s] += 1
        top_soft_skills = [{"name": k, "count": int(v)} for k, v in soft_counter.most_common(10)]

        return {
            "top_companies": companies_data,
            "top_tech_skills": top_tech_skills,
            "top_courses": top_courses,
            "top_soft_skills": top_soft_skills,
        }