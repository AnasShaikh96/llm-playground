from flask import Flask
from .api.moon import moon
from .api.sun import sun
from .api.internal import internal

def create_app():
    app = Flask(__name__)
    app.register_blueprint(moon, url_prefix=moon.url_prefix)
    app.register_blueprint(sun, url_prefix=sun.url_prefix)
    app.register_blueprint(internal, url_prefix=internal.url_prefix)

    return app