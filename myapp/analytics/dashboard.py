import statistics
from collections import Counter, defaultdict

# visits of the analysts to these pages are not usage of the search engine
NOT_ANALYSED_PATHS = ('/dashboard', '/stats', '/export', '/simulate')

DWELL_BINS = [(0, 5, '< 5 s'), (5, 15, '5-15 s'), (15, 30, '15-30 s'), (30, 60, '30-60 s'), (60, None, '> 60 s')]


def series(counter, top=None, sort_by_label=False):
    items = sorted(counter.items()) if sort_by_label else counter.most_common(top)
    return {'labels': [str(label) for label, _ in items], 'values': [value for _, value in items]}


def build_dashboard(data, corpus):
    # bots are not real users
    human = {sid for sid, s in data.sessions.items() if s.device != 'bot'}
    requests = [r for r in data.requests if r.session_id in human and not r.path.startswith(NOT_ANALYSED_PATHS)]
    sessions = [data.sessions[sid] for sid in {r.session_id for r in requests}]
    queries = [q for q in data.queries if q.session_id in human]
    clicks = [c for c in data.clicks if c.session_id in human]
    if not queries:
        return None

    result_clicks = [c for c in clicks if c.search_id is not None]
    details_clicks = [c for c in result_clicks if c.click_type == 'details']
    clicked_searches = {c.search_id for c in result_clicks}
    dwell_times = [c.dwell_time_s for c in details_clicks if c.dwell_time_s is not None]
    ranks = [c.rank for c in result_clicks if c.rank is not None]

    kpis = {
        'sessions': len(sessions),
        'queries': len(queries),
        'queries_per_session': round(len(queries) / len(sessions), 1) if sessions else 0,
        'ctr': round(100 * sum(q.search_id in clicked_searches for q in queries) / len(queries), 1),
        'zero_results': round(100 * sum(q.results_found == 0 for q in queries) / len(queries), 1),
        'or_mode': round(100 * sum(q.match_mode == 'or' for q in queries) / len(queries), 1),
        'mean_click_rank': round(statistics.mean(ranks), 1) if ranks else None,
        # median instead of mean: a user who leaves a product open for hours would distort the mean
        'median_dwell': round(statistics.median(dwell_times), 1) if dwell_times else None,
        'external_clicks': sum(c.click_type == 'external' for c in clicks),
    }

    dwell = Counter()
    for c in details_clicks:
        if c.dwell_time_s is None:
            dwell['no return'] += 1
            continue
        for low, high, label in DWELL_BINS:
            if c.dwell_time_s >= low and (high is None or c.dwell_time_s < high):
                dwell[label] += 1
    dwell_order = [label for _, _, label in DWELL_BINS] + ['no return']

    failed = Counter((q.query.lower(), q.results_found, q.match_mode if q.results_found else None) for q in queries
                     if q.results_found == 0 or q.match_mode == 'or')

    mission_sessions = defaultdict(set)
    mission_queries = defaultdict(list)
    for q in queries:
        mission_sessions[q.mission_id].add(q.session_id)
        mission_queries[q.mission_id].append(q.query)
    research = [m for m in mission_sessions if len(mission_sessions[m]) > 1]

    return {
        'kpis': kpis,
        'top_queries': series(Counter(q.query.lower() for q in queries), top=10),
        'top_terms': series(Counter(t for q in queries for t in q.terms), top=10),
        'query_length': series(Counter(q.num_terms for q in queries), sort_by_label=True),
        'methods': series(Counter(q.method for q in queries)),
        'failed_queries': [{'query': query, 'results': found, 'mode': mode, 'count': count}
                           for (query, found, mode), count in failed.most_common(10)],
        'clicks_by_rank': {'labels': [str(r) for r in range(1, 21)],
                           'values': [Counter(ranks)[r] for r in range(1, 21)]},
        'dwell': {'labels': dwell_order, 'values': [dwell[label] for label in dwell_order]},
        'top_documents': [{'pid': pid, 'title': corpus[pid].title if pid in corpus else pid, 'clicks': count}
                          for pid, count in Counter(c.pid for c in details_clicks).most_common(10)],
        'browsers': series(Counter(s.browser or 'unknown' for s in sessions)),
        'os': series(Counter(s.os or 'unknown' for s in sessions)),
        'devices': series(Counter(s.device or 'unknown' for s in sessions)),
        'hours': {'labels': [f'{h}h' for h in range(24)],
                  'values': [Counter(r.timestamp.hour for r in requests)[h] for h in range(24)]},
        'countries': series(Counter(s.country for s in sessions if s.country), top=10),
        'missions': {
            'total': len(mission_sessions),
            'research': len(research),
            'queries_per_mission': round(len(queries) / len(mission_sessions), 1),
            'research_examples': [{'mission': m, 'sessions': len(mission_sessions[m]),
                                   'queries': mission_queries[m]} for m in research[:5]],
        },
    }
