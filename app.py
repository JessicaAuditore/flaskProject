import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os, time, shutil
from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from inference import get_prediction
from pathlib import Path

app = Flask(__name__)
app.config.from_object('config')
app.secret_key = 'flask'
db = SQLAlchemy(app, use_native_unicode='utf8')


class User(db.Model):
    __tablename__ = 'user'
    __table_args__ = {'mysql_engine': 'InnoDB'}  # 支持事务操作和外键
    id = db.Column(db.INT, primary_key=True, autoincrement=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(128), nullable=False)


# db.create_all()


@app.route('/recognition', methods=['GET', 'POST'])
def recognition():
    if session.get('user') is None:
        return redirect('/')
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     return redirect(request.url)
        # img = request.files.get('file')
        # input_path = 'static/test/input_' + str(time.time()).replace('.', '') + str(img.filename)[
        #                                                                         str(img.filename).index('.'):]
        # if Path(input_path).exists():
        #     os.remove(input_path)
        # img.save(input_path)
        # output_path1 = get_prediction(input_path, 'deepR50')
        # output_path2 = get_prediction(input_path, 'mobileNet')

        output_path1 = 'static/test/label.png'
        output_path2 = 'static/test/label1.png'
        input_path = 'static/test/image.png'
        return render_template('result.html', input_path=input_path, output_path1=output_path1, output_path2=output_path2,
                               model1='ACA', model2='XJlEnc')
    else:
        return render_template('recognition.html')


@app.route('/register', methods=['POST'])
def register():
    name = request.values.get('name')
    password = request.values.get('password')
    password2 = request.values.get('password2')
    if password != password2:
        tip = '两次密码不同!'
    elif len(name) == 0 or len(password) == 0:
        tip = '用户名或密码不为空!'
    elif User.query.filter_by(name=name).first() is not None:
        tip = '用户已存在!'
    else:
        db.session.add(User(name=name, password=password))
        db.session.commit()
        tip = '注册成功!'
    return render_template('index.html', tip_register=tip)


@app.route('/login', methods=['POST'])
def login():
    name = request.values.get('name')
    password = request.values.get('password')
    if len(name) == 0 or len(password) == 0:
        tip = '用户名或密码不为空!'
    else:
        user = User.query.filter_by(name=name).first()
        if user is None:
            tip = '用户不存在!'
        elif user.password != password:
            tip = '密码错误!'
        else:
            session['user'] = user.name
            dir = 'static/person/' + user.name
            if not Path(dir).exists():
                os.mkdir('static/person/' + user.name)
            return redirect('/about.html')
    return render_template('index.html', tip_login=tip)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/index.html', methods=['GET'])
def index1():
    return render_template('index.html')


@app.route('/features.html', methods=['GET'])
def features():
    if session.get('user') is None:
        return redirect('/')
    return render_template('features.html')


@app.route('/about.html', methods=['GET'])
def contact():
    if session.get('user') is None:
        return redirect('/')
    return render_template('about.html')


@app.route('/dataset.html', methods=['GET'])
def dataset():
    if session.get('user') is None:
        return redirect('/')
    name = {'potsdam': 'ISPRS Potsdam', 'vaihingen': 'ISPRS Vaihingen'}
    introduction = {'potsdam': 'ISPRS Potsdam数据集采集自城市Postdam，图片大小为600×600像素，数据标签已手动划分为6种类别：建筑物、车辆、低矮植被、地面、树木和背景',
                    'vaihingen': 'ISPRS Vaihingen数据集采集自城市Vaihingen，图片大小为512×512像素，数据标签已手动划分为6种类别：建筑物、车辆、低矮植被、地面、树木和背景'}
    dataset = request.args.get('dataset')
    path = 'static/database/' + str(dataset)
    thing = request.args.get('thing')
    if thing == 'image':
        imageList = [path + '/images/' + i for i in os.listdir(path + '/images')]
    elif thing == 'mask':
        imageList = [path + '/masks/' + i for i in os.listdir(path + '/masks')]
    return render_template('dataset.html', dataset=dataset, name=name[dataset], imageList=imageList,
                           introduction=introduction[dataset])


@app.route('/person.html', methods=['GET'])
def person():
    if session.get('user') is None:
        return redirect('/')
    path = 'static/person/' + session.get('user')
    files = os.listdir(path)
    imageList = []
    for orgin in [file for file in files if 'input' in file]:
        output = orgin.replace('input', 'output')
        deepR50 = output.replace('.jpg', '_deepR50.jpg')
        mobileNet = output.replace('.jpg', '_mobileNet.jpg')
        imageList.append(
            {'origin': path + '/' + orgin, 'label1': path + '/' + deepR50, 'model1': 'ACA', 'label2': path + '/' + mobileNet, 'model2': 'XJlEnc'})
    return render_template('person.html', imageList=imageList)


@app.route('/save', methods=['GET'])
def save():
    if session.get('user') is None:
        return redirect('/')
    input_path = request.args.get('input_path')
    output_path1 = request.args.get('output_path1')
    output_path2 = request.args.get('output_path2')
    shutil.copy(input_path, 'static/person/' + session.get('user'))
    shutil.copy(output_path1, 'static/person/' + session.get('user'))
    shutil.copy(output_path2, 'static/person/' + session.get('user'))
    return render_template('recognition.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
