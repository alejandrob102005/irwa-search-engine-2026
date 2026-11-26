import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request, redirect, url_for, abort

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from project_progress.part_4.web_search import RANKING_METHODS
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome", methods=RANKING_METHODS)


# Part 4: GET is also accepted, so the results page has its own URL (/search?search-query=...)
# and the browser "back" button works without re-sending a form
@app.route('/search', methods=['GET', 'POST'])
def search_form_post():

    search_query = request.values.get('search-query', '').strip()
    method = request.values.get('method', 'custom')
    only_available = request.values.get('only-available') == 'on'
    if not search_query:
        return redirect(url_for('index'))

    session['last_search_query'] = search_query
    # used by the "back to results" link of the details page
    session['last_search_url'] = url_for('search_form_post', **{'search-query': search_query, 'method': method},
                                         **({'only-available': 'on'} if only_available else {}))

    search_id = analytics_data.save_query_terms(search_query)

    output = search_engine.search(search_query, search_id, corpus, method=method, top_k=20,
                                  only_available=only_available)

    # generate RAG response based on user query and retrieved results (only if there are results)
    rag_response = None
    if output.results:
        rag_response = rag_generator.generate_response(search_query, output.results)
        print("RAG response:", rag_response)

    session['last_found_count'] = output.total_found

    print(session)

    return render_template('results.html', page_title="Results", output=output, results_list=output.results,
                           found_counter=output.total_found, rag_response=rag_response,
                           methods=RANKING_METHODS, only_available=only_available)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """Document details page: shows all the information of the product in the corpus."""
    print("doc details session: ")
    print(session)

    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid")
    if clicked_doc_id not in corpus:
        abort(404)
    print("click in id={} (search_id={}, rank={})".format(clicked_doc_id, request.args.get("search_id"),
                                                          request.args.get("rank")))

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
    return render_template('doc_details.html', page_title="Product details", doc=corpus[clicked_doc_id],
                           rank=request.args.get("rank"), back_url=session.get('last_search_url'))


@app.route('/stats', methods=['GET'])
def stats():
    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
