"""This is main module of backend app"""

from flask import Flask, render_template, jsonify, abort, request
from autocorrect import spell
import nltk
from search_backend import Search
from search_backend.process_documents import index_files


def setup_environment():
    """Download required resources."""

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')
    index_files()


app = Flask(__name__)
setup_environment()

@app.route("/")
def root():
    return render_template('index.html')

search_global = Search()


@app.route('/search', methods=["POST"])
def search():
    json = request.get_json()

    if not json or json.get('query') is None:
        abort(400)

    query = json['query'].strip()

    return jsonify(
        results=search_global.search(query)
    )


@app.route('/spellCheck', methods=["POST"])
def spell_check():

    json = request.get_json()

    if not json or json.get('query') is None:
        abort(400)

    query = json['query'].strip()
    return jsonify(
        results=spell(query)
    )

if __name__ == "__main__":
    app.run()
