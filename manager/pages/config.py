import os
import shutil
import subprocess
import psutil
from flask import Blueprint, render_template, abort, Response, request, redirect, url_for
from jinja2 import TemplateNotFound
from ..config_params import ConfigParams
from ..db.sqlite_engine import db_session
from ..db.sql_config_model import Config
from ..env_vars import PATH_DIR, PATH_STANDARD_CONFIG

proc = None
cfg_params = ConfigParams()
config_page = Blueprint('config_page', __name__)


@config_page.route('/', defaults={'id': 0}, methods=['GET'])
@config_page.route('/<string:id>', methods=['GET', 'POST'])
def config(id):
    try:
        if id == 0: raise Exception('No config found')
        cfg = Config.query.filter_by(id=id).first()                             # Query for config with id
        path = cfg.path + cfg.name + '.yaml'                                    # Build complete path to yaml file

        if request.method == 'GET':
            res = cfg_params.load_yaml_config(path= path)                                                  
            return render_template('config.html', data= res + [id])             # Append uid to result for html href
                
        if request.method == 'POST':
            res = cfg_params.save_yaml_config(
                new_params = request.form.items(), 
                path= path)                                                     # path to save is defined in cfg_params from load_yaml
                                                 
            return render_template('config.html', data = res + [id])            # Append uid to result

    except Exception as ex:                                                     # TemplateNotFound:
        abort(Response('<h1>ERROR</h1>' + str(ex)))


@config_page.route('/start/<string:id>', methods=['GET'])
def start(id):

    global proc

    cfg = Config.query.filter_by(id=id).first()

    runner = 'run-ppo'
    config = cfg.name + '.yaml'         
    working_dir = cfg.path      

    proc = subprocess.Popen(['cmd'], creationflags=subprocess.CREATE_NEW_CONSOLE,
                            stdin=subprocess.PIPE, universal_newlines=True, bufsize=0)

    proc.stdin.write("  conda activate tf20rc\n \
                        python main.py --runner=%s --working_dir=%s --config=%s\n \
                        exit" % (runner, working_dir, config))
    proc.stdin.flush()
    proc.stdin.close()

    cfg_params.running_config = config

    return render_template('commands.html', data={"path": cfg_params.running_config})


@config_page.route('/stop')
def stop():

    global proc

    if proc is not None:
        kill(proc.pid)
        proc = None

    cfg_params.running_config = ""

    return redirect(url_for('main_page.show'))


def kill(proc_pid):

    process = psutil.Process(proc_pid)

    for proc in process.children(recursive=True):
        proc.kill()

    process.kill()
