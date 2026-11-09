from flask import Flask, render_template, request, jsonify
from dashboard_service import DashboardService
from vector_search import InternshipSearchEngine

app = Flask(__name__)

# Initialize services on backend startup
dashboard_service = DashboardService('database.csv')
search_engine = InternshipSearchEngine('database.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """REST API endpoint to feed analytics dashboard."""
    summary = dashboard_service.get_summary_stats()
    charts = dashboard_service.get_chart_data()
    return jsonify({"summary": summary, "charts": charts})

@app.route('/api/recommend', methods=['POST'])
def recommend_internships():
    """REST API endpoint for NLP search and structured recommendation."""
    data = request.get_json() or {}
    query = data.get('query', '')
    major = data.get('major', '')
    tech_skill = data.get('tech_skill', '')
    mode = data.get('mode', 'nlp')  # 'nlp' or 'filter'

    if mode == 'filter' or (major or tech_skill):
        results = search_engine.structured_filter(major=major, tech_skill=tech_skill)
    else:
        results = search_engine.search_nlp(query=query)

    return jsonify({"count": len(results), "results": results})

if __name__ == '__main__':
    print("STAT 4710 Web App is running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)