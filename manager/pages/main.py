import os
import shutil
from flask import Blueprint, render_template, abort, Response, request, redirect, url_for
from jinja2 import TemplateNotFound
from sqlalchemy import exc
from ..db.sqlite_engine import db_session
from ..db.sql_config_model import Config
from ..env_vars import PATH_DIR, PATH_STANDARD_CONFIG


main_page = Blueprint('main_page', __name__)


@main_page.route('/', defaults={'page': 'main'}, methods=['GET', 'POST'])
@main_page.route('/<page>')
def show(page):
    try:
        if request.method == 'GET':
            try:
                db_session.rollback()
                res = Config.query.all()                        # Get all configs from db
                
                return render_template('%s.html' % page, data = [res])

            except Exception as ex:
                print("ERROR >>" + str(ex))
                return {'Status': 'ERROR'}

        if request.method == 'POST':

            cfg_name = request.form['config_name']

            try:
                path = PATH_DIR + cfg_name + '/'

                cfg = Config(cfg_name, path)                    # insert in db with unique cfg_name
                db_session.add(cfg)
                db_session.commit()

                if not os.path.exists(path):                    # Make Dir and copy example.yaml to it with cfg_name.yaml
                    os.makedirs(path)
                    shutil.copy(PATH_STANDARD_CONFIG, path + cfg_name + '.yaml')
                else:
                    raise Exception ('Path already exists')

                return redirect(url_for('main_page.show'))

            except exc.SQLAlchemyError as ex:
                print("ERROR >> " + str(ex))
                return {'Status': 'ERROR ' + str(ex)}

    except Exception as ex:                                     # TemplateNotFound:
        abort(Response('Page not found'))


@main_page.route('/delete/<string:id>')
def delete(id):
    
    try:
        cfg_to_delete = Config.query.filter_by(id=id).first()   # Find config by uid and delete
        db_session.delete(cfg_to_delete)
        db_session.commit()

        if cfg_to_delete.path == './__WORKING_DIRS__//':        # Check not to delete the main path
            return {'Status': 'ERROR'}

        shutil.rmtree(cfg_to_delete.path)

        return redirect(url_for('main_page.show'))              # Delete folder with data

    except exc.SQLAlchemyError as ex:
            print("ERROR >> " + str(ex))
            return {'Status': 'ERROR ' + str(ex)}
