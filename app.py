from flask import Flask, render_template, request
from nova_core import run_nova

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/try', methods=["GET", "POST"])
def try_nova():
    code = ''
    result = ''
    error = ''
    if request.method == "POST":
        code = request.form.get("code", "")
        code = code.replace('\r\n', '\n').strip()
        code = "\n".join([line.strip() for line in code.splitlines() if line.strip() != ""])
        output, err = run_nova("<stdin>", code)
        result = str(output) if output else ''
        error = str(err) if err else ''
    return render_template("try.html", code=code, result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
