from flask import Flask, render_template
from flask_migrate import Migrate

migrate = Migrate()

def create_app():

    app = Flask(__name__)

    # configuracion app
    app.config.from_mapping(
        DEBUG = True,
        SECRET_KEY = "dev",
        JWT_SECRET_KEY="clave_jwt"
    )
    
    


    from . import home
    app.register_blueprint(home.bp)

    @app.route("/")
    def index():
        return render_template("index.html")
    
    
    return app