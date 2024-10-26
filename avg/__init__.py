from flask import Flask


def create_app():

    app = Flask(__name__)

    app.config.from_mapping(
        DEBUG = True,
        SECRET_KEY = "dev",
        JWT_SECRET_KEY="clave_jwt"
    )

    from . import modelo_lineal
    app.register_blueprint(modelo_lineal.bp)

    return app


