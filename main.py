## Author  Biju Subrahmanian ###

import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response

ALLOWED_EXTENSIONS = set(['mp4','avi','mov'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/uploadvideo/')
def upload_form():
	return render_template('upload.html')

@app.route('/uploadvideo/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		itemtype=request.form['classtype']
		productlabel=request.form['productlabel']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File(s) successfully uploaded')
			flash('Video Analytics Job kicked on...')
			import AutoAnnotationVideoService as av
			
			import AnnotateWithAugmentation as aa
			av.processVideo(filename=file.filename,itemtype=itemtype,labelname=productlabel,frameheight=1080,framewidth=1920)
			if request.form['augyesno'] == 'yes' :
				print(app.config['UPLOAD_FOLDER'] + '/' + productlabel)
				aa.doAug (filepath=app.config['UPLOAD_FOLDER'] + '/' + productlabel ,flipyesno=request.form['augflipyesno'],zoomyesno=request.form['augzoomyesno'],sample=int(request.form['sample']))
				aa.annotateAugedFiles(filepath=app.config['UPLOAD_FOLDER'] + '/' + productlabel + '/output',itemtype=itemtype,labelname=productlabel,frameheight=1080,framewidth=1920)
				#aa.annotateAugedFiles(app.config['UPLOAD_FOLDER'] )
			return redirect('/uploadvideo/')

if __name__ == "__main__":
    app.run()