"""This is main module of backend app"""

from flask import Flask, render_template, jsonify, abort, request
from autocorrect import spell
import nltk
from search_backend import Search


nltk.download('stopwords')
app = Flask(__name__)


@app.route("/")
def root():
    return render_template('index.html')

search_global = Search()
index = 'sdf'


@app.route('/search', methods=["POST"])
def search():
    json = request.get_json()

    if not json or json.get('query') is None:
        abort(400)

    query = str(json['query']).strip()

    return jsonify(
        results=search_global.search(query)
    )


@app.route('/spellCheck', methods=["POST"])
def spell_check():

    json = request.get_json()

    if not json or json.get('query') is None:
        abort(400)

    query = str(json['query']).strip()
    return jsonify(
        results=spell(query)
    )

if __name__ == "__main__":
    app.run()
