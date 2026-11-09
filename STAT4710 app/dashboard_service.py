import pandas as pd
from collections import Counter
import re

class DashboardService:
    def __init__(self, data_path='database.csv'):
        self.df = pd.read_csv(data_path).fillna('')

    def get_summary_stats(self):
        """Metric cards summary data."""
        return {
            "total_records": len(self.df),
            "total_companies": int(self.df['company'].nunique()),
            "total_majors": int(self.df['major'].nunique()),
            "total_courses": int(self.df['course'].nunique())
        }

    def _tokenize_column(self, col_name: str, top_n: int = 10):
        words = []
        for item in self.df[col_name]:
            if not item:
                continue
            # Split by common delimiters (commas, semicolons, bullets)
            tokens = [t.strip().title() for t in re.split(r'[,;\n•/]', str(item)) if len(t.strip()) > 2]
            words.extend(tokens)
        counts = Counter(words).most_common(top_n)
        return [{"name": name, "count": count} for name, count in counts]

    def get_chart_data(self):
        """Top charts aggregation."""
        # Top 10 Companies
        top_companies = self.df['company'].value_counts().head(10).to_dict()
        companies_data = [{"name": k, "count": int(v)} for k, v in top_companies.items() if k]

        # Top 10 Technical Skills
        top_tech_skills = self._tokenize_column('technical skills', 10)

        # Top 10 Majors represented
        top_majors = self.df['major'].value_counts().head(10).to_dict()
        majors_data = [{"name": k, "count": int(v)} for k, v in top_majors.items() if k]

        return {
            "top_companies": companies_data,
            "top_tech_skills": top_tech_skills,
            "top_majors": majors_data
        }