import os
from flask import Flask, render_template, request, url_for, redirect, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from knowledge_base_search_system import KnowledgeBaseSearchSystem
from database import insert_document

app = Flask(__name__, template_folder='../templates', static_folder='../static')

UPLOAD_FOLDER = "/Users/andrewasher/XYGen_ai/ML/Knowledge-Based-Search-Retrieval-System-main/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the Knowledge-Based Search Retrieval System
kb_system = KnowledgeBaseSearchSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

def get_pdf_page_count(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        return len(reader.pages)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        file = request.files['file']
        report_type = request.form['report_type']
        notes = request.form['notes']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            page_count = get_pdf_page_count(file_path)

            # Load the uploaded document into the Knowledge-Based Search System
            kb_system.load_documents([file_path])
            print('Sucessfully completed the document processing')
            # print(kb_system.document_texts)
                
            # database processing
            insert_document(filename, report_type, notes, page_count, file_path)

            message = "Success"

            return render_template('upload.html', message = message)
        else:
            message = "Failed"

            return render_template('upload.html', message = message)

    return render_template('upload.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_term = request.form.get('search_term')
        if not search_term:
            return jsonify({'error': 'No search term provided'}), 400
        
        search_results, llm_response = kb_system.search_documents(search_term)

        return render_template('search.html', search_term = search_term, 
                               search_results = search_results, llm_response = llm_response)
    
    return render_template('search.html')

@app.route('/pdfchat', methods=['GET', 'POST'])
def pdfchat():
    if request.method == 'POST':
        question = request.form.get('question')
        if not question:
            return jsonify({'error': 'No search term provided'}), 400
        
        response = kb_system.qa_conversation(question)

        return render_template('pdfchat.html', question = question, response = response)
    
    return render_template('pdfchat.html')

if __name__ == '__main__':
    app.run(debug=True)