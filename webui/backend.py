from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/endpoint', methods=['POST'])
def process_json():
    data = request.get_json()  # Parse JSON data from the request body
    question = data['question']
    response = {
        'question': question,
        'answer' : [{'name':'Debayan Banerjee','url':'https://dblp.org/pid/213/7475'},{'name':'Mohnish Dubey','url':'https://dblp.org/pid/180/1858'},{'name':'Debanjan Chaudhuri', 'url': 'https://dblp.org/pid/213/7337' },{'name':'Jens Lehmann','url':'https://dblp.org/pid/71/4882'}],
        'entities': ['https://dblp.org/rec/conf/semweb/DubeyBCL18'],
        'sparql': '''select * where { <https://dblp.org/rec/conf/semweb/DubeyBCL18> <https://dblp.org/rdf/schema#authoredBy> ?o}'''
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run()
