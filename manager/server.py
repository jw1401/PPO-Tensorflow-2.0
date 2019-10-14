from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from .pages.config import config_page
from .pages.main import main_page
from .db.sqlite_engine import db_session
from .config_params import ConfigParams


cfg_params = ConfigParams()
app = Flask('Manager', template_folder='./manager/templates', static_folder="./manager/static")
app.register_blueprint(main_page, url_prefix='/main')
app.register_blueprint(config_page, url_prefix='/config')
CORS(app)


@app.route('/')
def index():
    return redirect(url_for('main_page.show'))


@app.route('/commands')
def commands():
    return render_template('commands.html', data={"path": cfg_params.running_config})


@app.teardown_appcontext
def shutdown_session(exception = None):
    db_session.remove()
    

# --------------------------------------------------- #
def run_server():
    app.run(host='127.0.0.1', port=8080, debug=False)
