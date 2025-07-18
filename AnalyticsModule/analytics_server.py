# analytics_server.py
from flask import Flask
from flask_cors import CORS
from analytics import analytics_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(analytics_bp)

@app.route('/')
def home():
    return "<h1>Analytics Microservice is Up 🚀</h1>"

if __name__ == '__main__':
    print("--- Starting Analytics Server on port 5002 ---")
    app.run(debug=True, port=5002)
