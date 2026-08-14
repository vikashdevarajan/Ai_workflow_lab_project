import os
import json
import threading
import queue
from flask import Flask, render_template, request, Response, jsonify
from llm_as_a_judge import Rewriter, Judge, refine_until_approved, DEFAULT_MODEL
import groq

app = Flask(__name__)

# Check API key on startup
api_key = os.environ.get("GROQ_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream', methods=['POST'])
def stream():
    data = request.json
    original_text = data.get('text', '')
    if not original_text:
        return jsonify({"error": "No text provided"}), 400

    q = queue.Queue()

    def run_refinement():
        try:
            if not api_key:
                q.put({"type": "error", "message": "GROQ_API_KEY is not set. Please set it in your environment."})
                return

            client = groq.Groq(api_key=api_key)
            rewriter = Rewriter(client, model=DEFAULT_MODEL)
            judge = Judge(client, model=DEFAULT_MODEL)

            def callback(progress_data):
                q.put(progress_data)

            refine_until_approved(
                rewriter=rewriter,
                judge=judge,
                original_text=original_text,
                max_rounds=5,
                n_judge_samples=3,
                verbose=False,
                callback=callback
            )
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put({"type": "done"})

    thread = threading.Thread(target=run_refinement)
    thread.start()

    def generate():
        while True:
            item = q.get()
            if item.get("type") == "done":
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
