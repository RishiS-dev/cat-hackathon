# incident_logging/app.py
from flask import Flask
from flask_cors import CORS # Import CORS
import routes

def create_app():
    """Creates and configures a Flask application."""
    app = Flask(__name__)
    
    # --- ADD CORS SUPPORT ---
    # This allows your frontend (on localhost:8080) to make API calls
    # to this server (on localhost:5003) without being blocked by the browser.
    CORS(app)
    
    # Register the routes from routes.py
    app.register_blueprint(routes.bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    # Ensure it runs on port 5003 as planned
    app.run(host='0.0.0.0', port=5003, debug=True)