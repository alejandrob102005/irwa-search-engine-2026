# Part 4: RAG, User Interface, and Web Analytics

## Contents
- `web_search.py`: search engine of the web app. `WebSearchEngine` builds the Part 3 index once at start-up, and
  `search()` receives the query and returns the ranked products (Custom, BM25 or TF-IDF).
- `optimization_check.py`: checks that the faster Part 4 BM25 / custom ranking gives the same ranking as Part 3 and
  compares their times.
- `report_user_interface.md`: explanation of the User Interface part for the report.

The template files `web_app.py`, `myapp/search/search_engine.py`, `templates/` and `static/styles/custom.css` were
adapted to use this search engine.

## Run the web application
Python version used for verification: **3.11**. From the repository root:

```bash
pip install flask httpagentparser groq python-dotenv altair pydantic faker
copy .env.example .env        # Linux/Mac: cp .env.example .env
python web_app.py
```

Open [http://127.0.0.1:8088/](http://127.0.0.1:8088/) when `Running on http://127.0.0.1:8088` appears (the first start
takes some seconds). Add your `GROQ_API_KEY` in `.env` to get the AI summary; without it the search still works.

In the search page you can choose the **ranking method** and filter **only in stock** products.

## Check the optimization

```bash
python project_progress/part_4/optimization_check.py
```
