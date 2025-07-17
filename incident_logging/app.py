from flask import Flask
import routes

def create_app():
    """Creates and configures a Flask application."""
    app = Flask(__name__)
    
    # Register the routes from routes.py
    app.register_blueprint(routes.bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001) # Using port 5001 to avoid conflicts