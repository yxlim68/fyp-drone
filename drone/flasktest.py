from flask import Flask, send_from_directory, render_template
import os

app = Flask(__name__)

@app.route('/snapshots/<filename>')
def download_snapshot(filename):
    return send_from_directory('C:\\xampp\\htdocs\\snapshots', filename)

@app.route('/')
def list_snapshots():
    snapshot_dir = 'C:\\xampp\\htdocs\\snapshots'
    snapshots = os.listdir(snapshot_dir)
    return render_template('index.html', snapshots=snapshots)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
