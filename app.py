from flask import Flask, request, jsonify, render_template
from recommendation_engine.recommendation_engine import RecommendationEngine

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/recommendations", methods=["POST"])
def get_recommendations():
    username = request.form["usernameRec"]
    # Load the recommendation engine
    engine = RecommendationEngine()

    # Get the recommendations for the user
    recommendations = engine.get_recommendations_by_sentiment(username)
    try:
        recommended_products = recommendations['name'].values.tolist()[:5]
        return render_template("index.html", recommendations= recommended_products)
    except:
        failure_str = "Username doesn't exist, please try again with a different username"
        return render_template("index.html", failure= failure_str)

if __name__ == "__main__":
  app.run(debug = False, host="0.0.0.0", port = 5000)
