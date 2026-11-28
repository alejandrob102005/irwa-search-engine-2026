import os
import time
import uuid
from datetime import datetime, timedelta
import json
from json import JSONEncoder

from flask import Flask, render_template, session, g, Response
from flask import request, redirect, url_for, abort

from myapp.analytics.analytics_data import AnalyticsData, parse_user_agent, query_terms, get_location
from myapp.analytics.dashboard import build_dashboard
from myapp.analytics.simulation import simulate_traffic
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
data_file_path = os.getenv("DATA_FILE_PATH") or "data/fashion_products_dataset.json"
file_path = os.path.join(path, data_file_path)
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Part 4 - Web Analytics: every request is saved in the Request table and every visitor
# gets a row in the Session table. CSS/images and the favicon are not tracked (noise).
VISITOR_COOKIE = 'irwa_visitor_id'
NOT_TRACKED = ('/static/', '/favicon.ico', '/returned')
# physical session: it ends after 30 minutes without requests (same value as Google Analytics)
SESSION_TIMEOUT = timedelta(minutes=30)


@app.before_request
def start_tracking():
    if request.path.startswith(NOT_TRACKED):
        return
    g.start_time = time.perf_counter()
    g.visitor_id = request.cookies.get(VISITOR_COOKIE) or uuid.uuid4().hex

    session_id = session.get('session_id')
    current = analytics_data.sessions.get(session_id)
    if current and datetime.now() - current.last_seen > SESSION_TIMEOUT:
        current = None
        # coming back after the timeout is not a real dwell time
        session.pop('last_click_id', None)
        session.pop('last_click_time', None)
    # new visitor, inactive session, or the server was restarted and the session is no longer in memory
    if current is None:
        session_id = uuid.uuid4().hex
        session['session_id'] = session_id
        user_agent = request.headers.get('User-Agent')
        browser, os_name, device = parse_user_agent(user_agent)
        country, city = get_location(request.remote_addr)
        analytics_data.save_session(session_id, g.visitor_id, ip=request.remote_addr, user_agent=user_agent,
                                    browser=browser, os=os_name, device=device, country=country, city=city)
    g.session_id = session_id


@app.after_request
def save_request(response):
    if 'session_id' not in g:
        return response
    response_ms = (time.perf_counter() - g.start_time) * 1000
    analytics_data.save_request(g.session_id, request.method, request.path, status_code=response.status_code,
                                response_ms=response_ms, referrer=request.referrer)
    if request.cookies.get(VISITOR_COOKIE) != g.visitor_id:
        response.set_cookie(VISITOR_COOKIE, g.visitor_id, max_age=365 * 24 * 3600, httponly=True, samesite='Lax')
    return response


# Home URL "/"
@app.route('/')
def index():
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

    # coming back to the results of a previous search is not a new query: either the
    # "back to results" link (sid) or the same search as the pending click (browser back button)
    previous = analytics_data.get_query(request.values.get('sid', type=int))
    if not (previous and previous.session_id == g.get('session_id') and previous.query == search_query):
        previous = pending_click_query(search_query, method, only_available)
    is_new_query = previous is None
    if is_new_query:
        # the user did not come back to the results of the clicked product, so it has no dwell time
        session.pop('last_click_id', None)
        session.pop('last_click_time', None)
        processed_terms = search_engine.engine.preprocessor.preprocess_text(search_query)
        search_id = analytics_data.save_query(g.get('session_id'), search_query, query_terms(search_query),
                                              processed_terms=processed_terms, method=method,
                                              only_available=only_available)
    else:
        search_id = previous.search_id
        close_pending_click()

    # used by the "back to results" link of the details page
    session['last_search_url'] = url_for('search_form_post', **{'search-query': search_query, 'method': method},
                                         **({'only-available': 'on'} if only_available else {}), sid=search_id)

    output = search_engine.search(search_query, search_id, corpus, method=method, top_k=20,
                                  only_available=only_available)
    if is_new_query:
        analytics_data.set_query_results(search_id, output.total_found, output.match_mode)

    # generate RAG response based on user query and retrieved results (only if there are results)
    rag_response = None
    if output.results:
        rag_response = rag_generator.generate_response(search_query, output.results)
        print("RAG response:", rag_response)

    session['last_found_count'] = output.total_found

    print(session)

    return render_template('results.html', page_title="Results", output=output, results_list=output.results,
                           found_counter=output.total_found, rag_response=rag_response,
                           methods=RANKING_METHODS, only_available=only_available, search_id=search_id)


def pending_click_query(search_query, method, only_available):
    click = analytics_data.get_click(session.get('last_click_id'))
    query = analytics_data.get_query(click.search_id) if click else None
    if (query and query.session_id == g.get('session_id') and query.query == search_query
            and query.method == method and query.only_available == only_available):
        return query
    return None


def close_pending_click():
    click = analytics_data.get_click(session.pop('last_click_id', None))
    click_time = session.pop('last_click_time', None)
    # only the first return counts, a later one does not overwrite the dwell time
    if click and click_time is not None and click.dwell_time_s is None:
        analytics_data.set_dwell_time(click.click_id, round(time.time() - click_time, 1))


# called by the JavaScript of results.html when the browser shows the results page
# from its cache (back button), because in that case no request reaches the server
@app.route('/returned', methods=['POST'])
def returned_to_results():
    close_pending_click()
    return '', 204


def read_click_args():
    # search_id and rank come from the result links; a details page opened without a
    # search (or with wrong values) is saved with search_id and rank empty
    search_id = request.args.get('search_id', type=int)
    rank = request.args.get('rank', type=int)
    if analytics_data.get_query(search_id) is None:
        return None, None
    return search_id, rank


@app.route('/doc_details', methods=['GET'])
def doc_details():
    clicked_doc_id = request.args.get("pid")
    if clicked_doc_id not in corpus:
        abort(404)

    search_id, rank = read_click_args()
    click_id = analytics_data.save_click(g.get('session_id'), search_id, clicked_doc_id, rank)
    # used in step 4 to compute the dwell time when the user goes back to the results
    session['last_click_id'] = click_id
    session['last_click_time'] = time.time()

    return render_template('doc_details.html', page_title="Product details", doc=corpus[clicked_doc_id],
                           rank=rank, search_id=search_id, back_url=session.get('last_search_url'))


# the link to the original website goes through this route, so the click is saved before leaving the site
@app.route('/go', methods=['GET'])
def go_to_product():
    pid = request.args.get('pid')
    if pid not in corpus or not corpus[pid].url:
        abort(404)
    search_id, rank = read_click_args()
    analytics_data.save_click(g.get('session_id'), search_id, pid, rank, click_type='external')
    return redirect(corpus[pid].url)


@app.route('/stats', methods=['GET'])
def stats():
    docs = []
    for doc_id, count in analytics_data.clicks_per_document().items():
        row: Document = corpus[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    simulated_sessions = sum(s.simulated for s in analytics_data.sessions.values())
    return render_template('dashboard.html', page_title="Dashboard", data=build_dashboard(analytics_data, corpus),
                           simulated_sessions=simulated_sessions)


@app.route('/simulate', methods=['POST'])
def simulate():
    simulate_traffic(analytics_data, search_engine.engine, corpus)
    return redirect(url_for('dashboard'))


@app.route('/export', methods=['GET'])
def export():
    content = json.dumps(analytics_data.to_dict(), indent=2, default=str)
    return Response(content, mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=analytics_data.json'})


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
