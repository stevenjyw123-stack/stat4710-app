import re
from collections import Counter, defaultdict
import pandas as pd


BASE_MAJORS = [
    "Actuarial Science",
    "Risk Management",
    "Statistics",
    "Applied AI",
    "Decision Analytics",
    "Data Analytics",
]


class DashboardService:
    def __init__(self, data_path="database.csv"):
        self.df = pd.read_csv(data_path).fillna("")
        self.df["_base_majors"] = self.df["major"].apply(self._extract_base_majors)
        self.df["_courses_list"] = self.df["courses mentioned"].apply(self._extract_courses)
        self.df["_skills_list"] = self.df.apply(self._extract_skills, axis=1)
        self.df["_soft_skills_list"] = self.df["soft skills"].apply(self._extract_soft_skills)
        self.df["_focus_areas"] = self.df.apply(self._extract_focus_areas, axis=1)

    @staticmethod
    def _extract_base_majors(raw_major: str) -> list[str]:
        text = str(raw_major or "").strip()
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"[/,&]+", text) if p.strip()]
        found = []
        alias_map = {
            "actuarial": "Actuarial Science",
            "risk": "Risk Management",
            "statistic": "Statistics",
            "applied ai": "Applied AI",
            "ai": "Applied AI",
            "decision": "Decision Analytics",
            "data analytics": "Data Analytics",
        }
        for part in parts:
            low = part.lower()
            matched = None
            for k, canonical in alias_map.items():
                if k in low:
                    matched = canonical
                    break
            if matched and matched not in found:
                found.append(matched)
            elif not matched and part not in found:
                found.append(part)
        return found

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
            ("VBA", r"\bvba\b"),
            ("Python", r"\bpython\b"),
            ("SQL", r"\bsql\b"),
            ("Prophet", r"\bprophet\b"),
            ("IFRS 17", r"\bifrs\s*17\b"),
            ("R", r"(?<![a-zA-Z0-9_])r(?![a-zA-Z0-9_])"),
            ("SAS", r"\bsas\b"),
            ("Power BI", r"\bpower\s*bi\b"),
            ("AI Agent", r"\bagent(?:s)?\b|\brag\b|\bllm\b"),
            ("Tableau", r"\btableau\b"),
        ]
        return [label for label, pat in skill_patterns if re.search(pat, blob, flags=re.IGNORECASE)]

    @staticmethod
    def _extract_soft_skills(text: str) -> list[str]:
        blob = str(text or "").lower()
        if not blob.strip():
            return []
        soft_patterns = [
            ("Communication", r"communicat"),
            ("Teamwork & Collaboration", r"teamwork|collaborat|teammate"),
            ("Time Management", r"time management|deadline|prioritiz"),
            ("Adaptability & Learning", r"adapt|fast learn|self-learn|independ"),
            ("Stakeholder / Client Mgmt", r"client|stakeholder|cross-functional"),
            ("Attention to Detail", r"detail|accuracy|careful"),
            ("Presentation", r"presentat"),
            ("Problem Solving", r"problem[- ]solv|troubleshoot"),
            ("Project Management", r"project management|coordinat"),
            ("Documentation & Reporting", r"document|report writing"),
        ]
        return [label for label, pat in soft_patterns if re.search(pat, blob)]

    @staticmethod
    def _extract_focus_areas(row) -> list[str]:
        blob = f"{row.get('job title', '')} {row.get('job description', '')} {row.get('job duties', '')}".lower()
        focus_patterns = [
            ("Valuation & Pricing", r"valuat|pricing|reserving|actuarial"),
            ("Data Analysis & Analytics", r"data analy|analytics|insights|dashboard"),
            ("Model & Workflow Automation", r"automat|macro|vba|pipeline|etl"),
            ("IFRS 17 & Regulatory", r"ifrs\s*17|regulat|compliance|audit|risk"),
            ("Forecasting & Modeling", r"forecast|model|time series|prophet|regression"),
            ("AI & LLM Applications", r"\bai\b|\bllm\b|\brag\b|machine learning|agent"),
            ("Reporting & Visualization", r"report|visuali|power\s*bi|tableau|excel"),
        ]
        return [label for label, pat in focus_patterns if re.search(pat, blob)]

    @staticmethod
    def _clean_company_name(raw_company: str) -> str:
        name = str(raw_company or "").strip()
        if not name:
            return ""
        # Fix common truncated prefix in raw dataset
        if name.lower().startswith("d actuarial science"):
            return "Department of Statistics and Actuarial Science, HKU"
        return name

    def get_summary_stats(self) -> dict:
        """Metric cards summary data with clean base-major counts."""
        companies_clean = [self._clean_company_name(c) for c in self.df["company"]]
        companies_non_empty = [c for c in companies_clean if c]

        base_major_counter = Counter()
        unspecified_count = 0
        for majors in self.df["_base_majors"]:
            if not majors:
                unspecified_count += 1
            else:
                for m in majors:
                    base_major_counter[m] += 1

        majors_breakdown = [
            {"name": name, "count": int(cnt)}
            for name, cnt in base_major_counter.most_common()
        ]
        if unspecified_count > 0:
            majors_breakdown.append({"name": "Unspecified Major", "count": int(unspecified_count)})

        course_codes_clean = [
            str(c).strip() for c in self.df["course"] if str(c).strip()
        ]

        return {
            "year": "2025",
            "total_records": int(len(self.df)),
            "total_companies": int(len(set(companies_non_empty))),
            "named_records": int(len(companies_non_empty)),
            "total_majors": int(len(base_major_counter)),
            "total_courses": int(len(set(course_codes_clean))),
            "majors_breakdown": majors_breakdown,
            "available_majors": BASE_MAJORS,
        }

    def get_chart_data(self) -> dict:
        """Charts data for dashboard summary tab, including 3-variable and advanced charts."""
        companies_series = self.df["company"].apply(self._clean_company_name)
        top_companies_s = companies_series[companies_series != ""].value_counts().head(10)
        companies_data = [{"name": str(k), "count": int(v)} for k, v in top_companies_s.items()]

        skill_counter = Counter()
        for skills in self.df["_skills_list"]:
            for s in skills:
                skill_counter[s] += 1
        top_tech_skills = [{"name": k, "count": int(v)} for k, v in skill_counter.most_common(10)]

        course_counter = Counter()
        for courses in self.df["_courses_list"]:
            for c in courses:
                course_counter[c] += 1
        top_courses = [{"name": k, "count": int(v)} for k, v in course_counter.most_common(10)]

        soft_counter = Counter()
        for s_list in self.df["_soft_skills_list"]:
            for s in s_list:
                soft_counter[s] += 1
        top_soft_skills = [{"name": k, "count": int(v)} for k, v in soft_counter.most_common(10)]

        focus_counter = Counter()
        for f_list in self.df["_focus_areas"]:
            for f in f_list:
                focus_counter[f] += 1
        focus_areas = [{"name": k, "count": int(v)} for k, v in focus_counter.most_common(8)]

        # Semester x Course Code breakdown (3 dimensions: Semester, Course Code, Count)
        semesters = ["Semester 1", "Semester 2", "Summer"]
        course_codes = ["STAT4767", "STAT4766", "APAI4766"]
        sem_course_matrix = defaultdict(int)
        for _, row in self.df.iterrows():
            sem = str(row.get("semester", "")).strip() or "Other"
            cc = str(row.get("course", "")).strip() or "Other"
            if sem not in semesters:
                sem = "Other"
            if cc not in course_codes:
                cc = "Other"
            sem_course_matrix[(sem, cc)] += 1

        active_sems = [s for s in semesters + ["Other"] if any(sem_course_matrix[(s, c)] > 0 for c in course_codes + ["Other"])]
        active_ccs = [c for c in course_codes + ["Other"] if any(sem_course_matrix[(s, c)] > 0 for s in active_sems)]
        semester_course_chart = {
            "labels": active_sems,
            "datasets": [
                {
                    "label": cc,
                    "data": [int(sem_course_matrix[(s, cc)]) for s in active_sems],
                }
                for cc in active_ccs
            ],
        }

        # 3-Variable Grouped Bar: Top 5 Host Companies x Top 5 Technical Skills
        top5_companies = [c["name"] for c in companies_data[:5]]
        top5_skills = [s["name"] for s in top_tech_skills[:5]]
        comp_skill_counts = defaultdict(int)
        for idx, row in self.df.iterrows():
            comp = self._clean_company_name(row.get("company", ""))
            if comp in top5_companies:
                for sk in self.df.loc[idx, "_skills_list"]:
                    if sk in top5_skills:
                        comp_skill_counts[(comp, sk)] += 1

        company_skill_3var = {
            "labels": top5_companies,
            "datasets": [
                {
                    "label": sk,
                    "data": [int(comp_skill_counts[(comp, sk)]) for comp in top5_companies],
                }
                for sk in top5_skills
            ],
        }

        # 3-Variable Grouped Bar: Top 5 Host Companies x Base Majors
        major_counter = Counter()
        for m_list in self.df["_base_majors"]:
            for m in m_list:
                major_counter[m] += 1
        top5_majors = [m for m, _ in major_counter.most_common(5)]
        comp_major_counts = defaultdict(int)
        for idx, row in self.df.iterrows():
            comp = self._clean_company_name(row.get("company", ""))
            if comp in top5_companies:
                for mj in self.df.loc[idx, "_base_majors"]:
                    if mj in top5_majors:
                        comp_major_counts[(comp, mj)] += 1

        company_major_3var = {
            "labels": top5_companies,
            "datasets": [
                {
                    "label": mj,
                    "data": [int(comp_major_counts[(comp, mj)]) for comp in top5_companies],
                }
                for mj in top5_majors
            ],
        }

        # Radar Chart: Top 6 Soft Skills normalized to % of records with soft skills
        total_records = max(1, len(self.df))
        radar_top6 = top_soft_skills[:6]
        radar_chart = {
            "labels": [x["name"] for x in radar_top6],
            "values": [round(x["count"] * 100.0 / total_records, 1) for x in radar_top6],
        }

        # Treemap-style blocks: Base Majors share of dataset
        treemap_majors = [
            {
                "name": m,
                "count": int(cnt),
                "share": round(cnt * 100.0 / total_records, 1),
            }
            for m, cnt in major_counter.most_common()
        ]

        return {
            "top_companies": companies_data,
            "top_tech_skills": top_tech_skills,
            "top_courses": top_courses,
            "top_soft_skills": top_soft_skills,
            "focus_areas": focus_areas,
            "semester_course_chart": semester_course_chart,
            "company_skill_3var": company_skill_3var,
            "company_major_3var": company_major_3var,
            "radar_chart": radar_chart,
            "treemap_majors": treemap_majors,
        }