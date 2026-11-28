import math
import random
import uuid
from datetime import datetime, timedelta

from myapp.analytics.analytics_data import query_terms

# (browser, os, device), weight
CLIENTS = [(('Chrome', 'Windows', 'desktop'), 30), (('Chrome', 'Android', 'mobile'), 22),
           (('Safari', 'iOS', 'mobile'), 15), (('Edge', 'Windows', 'desktop'), 8),
           (('Safari', 'Macintosh', 'desktop'), 8), (('Firefox', 'Windows', 'desktop'), 7),
           (('Safari', 'iOS', 'tablet'), 5), (('Chrome', 'Linux', 'desktop'), 5)]

# the dataset comes from an Indian e-commerce website, so most visitors are from India
LOCATIONS = [(('India', 'Mumbai'), 25), (('India', 'Delhi'), 22), (('India', 'Bengaluru'), 18),
             (('India', 'Chennai'), 10), (('India', 'Kolkata'), 8), (('Spain', 'Barcelona'), 7),
             (('United States', 'New York'), 5), (('United Kingdom', 'London'), 5)]

# more activity in the evening than at night
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 5, 5, 6, 7, 8, 10, 11, 10, 7, 3]

# each goal is a sequence of reformulations of the same need (one mission); the last two
# start with a typo, so they give searches without results or in OR mode
GOALS = [
    ['men leather jacket', 'men black leather jacket', 'black leather jacket'],
    ['women kurta', 'women cotton printed kurta', 'women cotton kurta'],
    ['men slim fit shirt', 'men slim fit formal checkered shirt', 'men formal shirt'],
    ['women sweatshirt', 'women hooded winter sweatshirt', 'hooded sweatshirt'],
    ['cotton socks', 'cotton ankle socks pack', 'ankle socks'],
    ['men jeans', 'men slim jeans blue', 'blue jeans'],
    ['women track pants', 'women cotton track pants'],
    ['men tshirt', 'men round neck tshirt', 'men polo tshirt'],
    ['jaket', 'jacket'],
    ['leathr shoes men', 'men shoes'],
]
METHODS = [('custom', 70), ('bm25', 15), ('tfidf', 15)]


def choose(rng, options):
    values, weights = zip(*options)
    return rng.choices(values, weights=weights)[0]


def click_probability(rank):
    # position bias: users click much more on the first results
    return 0.5 / rank ** 1.2


def simulate_traffic(data, engine, corpus, n_visitors=60, days=30, seed=42):
    # the data is generated only once, pressing the button again would duplicate it
    if any(s.simulated for s in data.sessions.values()):
        return 0
    # fixed seed: the same data every time, so the numbers of the report can be reproduced
    rng = random.Random(seed)
    now = datetime.now().replace(second=0, microsecond=0)
    n_sessions = 0

    for v in range(n_visitors):
        visitor_id = uuid.UUID(int=rng.getrandbits(128)).hex
        client = choose(rng, CLIENTS)
        location = choose(rng, LOCATIONS)
        # 203.0.113.x is a range reserved for documentation, it is not a real user
        ip = f'203.0.113.{v + 1}'
        favourite_goal = rng.choice(GOALS)

        visits = rng.choices([1, 2, 3], weights=[55, 30, 15])[0]
        days_ago = sorted(rng.sample(range(1, days + 1), visits), reverse=True)
        for d in days_ago:
            start = (now - timedelta(days=d)).replace(hour=rng.choices(range(24), weights=HOUR_WEIGHTS)[0],
                                                      minute=rng.randint(0, 59))
            # returning visitors usually continue with the same need (research mission)
            goals = [favourite_goal] if rng.random() < 0.7 else []
            goals += rng.sample(GOALS, rng.randint(0 if goals else 1, 1))
            simulate_session(data, engine, corpus, rng, visitor_id, client, location, ip, start, goals)
            n_sessions += 1
    return n_sessions


def simulate_session(data, engine, corpus, rng, visitor_id, client, location, ip, start, goals):
    session_id = uuid.UUID(int=rng.getrandbits(128)).hex
    browser, os_name, device = client
    country, city = location
    data.save_session(session_id, visitor_id, ip=ip, browser=browser, os=os_name, device=device,
                      country=country, city=city, timestamp=start, simulated=True)
    t = start
    data.save_request(session_id, 'GET', '/', status_code=200, response_ms=rng.uniform(5, 20), timestamp=t)

    for goal in goals:
        method = choose(rng, METHODS)
        for query in goal[:rng.randint(1, len(goal))]:
            t += timedelta(seconds=rng.uniform(5, 40))
            output = engine.search(query, corpus, method=method, top_k=20)
            search_id = data.save_query(session_id, query, query_terms(query),
                                        processed_terms=engine.preprocessor.preprocess_text(query),
                                        method=method, timestamp=t)
            data.set_query_results(search_id, output.total_found, output.match_mode)
            data.save_request(session_id, 'GET', '/search', status_code=200,
                              response_ms=rng.uniform(30, 200), timestamp=t)

            clicks = 0
            for result in output.results:
                if clicks == 3 or rng.random() >= click_probability(result.rank):
                    continue
                clicks += 1
                t += timedelta(seconds=rng.uniform(2, 10))
                click_id = data.save_click(session_id, search_id, result.pid, result.rank, timestamp=t)
                data.save_request(session_id, 'GET', '/doc_details', status_code=200,
                                  response_ms=rng.uniform(5, 30), timestamp=t)
                if rng.random() < 0.2:
                    t += timedelta(seconds=rng.uniform(5, 30))
                    data.save_click(session_id, search_id, result.pid, result.rank, click_type='external',
                                    timestamp=t)
                    data.save_request(session_id, 'GET', '/go', status_code=302,
                                      response_ms=rng.uniform(1, 5), timestamp=t)
                if rng.random() < 0.15:
                    break
                # most dwell times between 5 and 60 seconds, a few very short or very long
                dwell = round(rng.lognormvariate(math.log(20), 1.0), 1)
                data.set_dwell_time(click_id, dwell)
                t += timedelta(seconds=dwell)
